"""Data preprocessing module for cleaning and normalizing datasets."""
import argparse
import logging
import pickle
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

MEDICAL_COLS = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

def load_data(file_path: str) -> pd.DataFrame:
    """Load dataset from a CSV file."""
    if not Path(file_path).exists():
        raise FileNotFoundError(f"The file {file_path} was not found.")  # noqa: EM102, TRY003
    data = pd.read_csv(file_path)
    logger.info(f"Data loaded successfully from {file_path}")
    return data

def perform_correlation_analysis(df: pd.DataFrame, output_path: str | None = None) -> None:
    """Perform correlation analysis to ensure high-quality input features."""
    plt.figure(figsize=(10, 8))
    correlation_matrix = df.corr(numeric_only=True)
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Feature Correlation Heatmap")

    if output_path:
        plt.savefig(output_path, bbox_inches="tight")
        logger.info(f"Correlation heatmap saved to {output_path}")
    else:
        plt.show()
    plt.close()

def _replace_zeros_with_nan(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    for col in cols:
        if col in df.columns:
            df[col] = df[col].replace(0, np.nan)
    return df

def fit_imputer_medians(x_train: pd.DataFrame, cols: list[str]) -> dict[str, float]:
    """Fit median values on training data for specified columns."""
    medians: dict[str, float] = {}
    for col in cols:
        if col in x_train.columns:
            med = float(x_train[col].median())
            medians[col] = med
    return medians

def apply_imputer_medians(x: pd.DataFrame, medians: dict[str, float]) -> pd.DataFrame:
    """Apply median imputation to specified columns."""
    x = x.copy()
    for col, med in medians.items():
        if col in x.columns:
            x[col] = x[col].fillna(med)
    return x

def preprocess_and_split(
        df: pd.DataFrame,
        target_col: str = "Outcome",
        test_size: float = 0.2,
        val_size: float = 0.2,
        make_val: bool = True,
        seed: int = 42,
    ) -> dict[str, Any]:
    """Split data into train and test sets."""
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset.")  # noqa: EM102, TRY003

    x = df.drop(columns=[target_col])
    y = df[target_col].astype(int)

    x_train_full, x_test, y_train_full, y_test = train_test_split(
        x, y, test_size=test_size, random_state=seed, stratify=y
    )

    if make_val:
        x_train, x_val, y_train, y_val = train_test_split(
            x_train_full,
            y_train_full,
            test_size=val_size,
            random_state=seed,
            stratify=y_train_full,
        )
    else:
        x_train, y_train = x_train_full, y_train_full
        x_val, y_val = None, None

    x_train = _replace_zeros_with_nan(x_train, MEDICAL_COLS)
    x_test = _replace_zeros_with_nan(x_test, MEDICAL_COLS)
    if make_val and x_val is not None:
        x_val = _replace_zeros_with_nan(x_val, MEDICAL_COLS)

    medians = fit_imputer_medians(x_train, MEDICAL_COLS)

    x_train = apply_imputer_medians(x_train, medians)
    x_test = apply_imputer_medians(x_test, medians)
    if make_val and x_val is not None:
        x_val = apply_imputer_medians(x_val, medians)

    scaler = StandardScaler()
    x_train_scaled = pd.DataFrame(scaler.fit_transform(x_train), columns=x_train.columns)
    x_test_scaled = pd.DataFrame(scaler.transform(x_test), columns=x_test.columns)
    if make_val and x_val is not None:
        x_val_scaled = pd.DataFrame(scaler.transform(x_val), columns=x_val.columns)
    else:
        x_val_scaled = None

    logger.info(
        "Completed preprocessing.\n"
        f"Train size: {len(x_train_scaled)} | "
        f"Val size: {len(x_val_scaled) if x_val_scaled is not None else 0} | "
        f"Test size: {len(x_test_scaled)}"
    )

    artifacts = {
        "columns": list(x_train.columns),
        "medians": medians,
        "scaler": scaler,
        "seed": seed,
        "target_col": target_col,
    }

    return {
        "x_train": x_train_scaled,
        "y_train": y_train.reset_index(drop=True),
        "x_val": x_val_scaled,
        "y_val": y_val.reset_index(drop=True) if y_val is not None else None,
        "x_test": x_test_scaled,
        "y_test": y_test.reset_index(drop=True),
        "artifacts": artifacts,
    }

def save_processed_data(bundle: dict[str, Any], output_dir: str) -> None:
    """Save processed data to specified output directory."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    bundle["x_train"].to_csv(output_dir / "x_train.csv", index=False)
    bundle["y_train"].to_csv(output_dir / "y_train.csv", index=False)
    bundle["x_test"].to_csv(output_dir / "x_test.csv", index=False)
    bundle["y_test"].to_csv(output_dir / "y_test.csv", index=False)

    if bundle["x_val"] is not None and bundle["y_val"] is not None:
        bundle["x_val"].to_csv(output_dir / "x_val.csv", index=False)
        bundle["y_val"].to_csv(output_dir / "y_val.csv", index=False)

    with (output_dir / "preprocess_artifacts.pkl").open("wb") as f:
        pickle.dump(bundle["artifacts"], f)

    logger.info(f"Processed data saved to {output_dir}")
    logger.info(f"Preprocessing artifacts saved to {output_dir / 'preprocess_artifacts.pkl'}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data Preprocessing Module")

    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to the raw CSV data file.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="../data/processed/",
        help="Output directory.",
    )
    parser.add_argument(
        "--target",
        type=str,
        default="Outcome",
        help="Target column name.",
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.2,
        help="Test set fraction.",
    )
    parser.add_argument(
        "--val_size",
        type=float,
        default=0.2,
        help="Validation set fraction from training data.",
    )
    parser.add_argument(
        "--no_val",
        action="store_true",
        help="If set, no validation set will be created.",
    )
    args = parser.parse_args()

    try:
        # Load
        data = load_data(args.input)
        output_dir = args.output if args.output.endswith("/") else args.output + "/"

        # Analyse
        perform_correlation_analysis(data, output_path=output_dir + "correlation_heatmap.png")

        # Preprocess
        bundle = preprocess_and_split(
            data,
            target_col=args.target,
            test_size=args.test_size,
            val_size=args.val_size,
            make_val=not args.no_val,
            seed=42,
        )

        # Save
        save_processed_data(bundle, output_dir)
    except Exception as e:
        logger.exception(f"An error occurred during preprocessing: {e}")
