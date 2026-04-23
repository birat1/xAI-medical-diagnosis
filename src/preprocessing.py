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
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

SYMBOLIC_DIR = Path("../data/symbolic/")
SYMBOLIC_DIR.mkdir(parents=True, exist_ok=True)
MEDICAL_COLS = ["glucose", "bloodpressure", "skinthickness", "insulin", "bmi"]

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

def split_train_test(
        df: pd.DataFrame,
        target_col: str = "outcome",
        test_size: float = 0.2,
        seed: int = 42,
    ) -> dict[str, Any]:
    """Split data into development and test sets for 5-fold CV."""
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset.")  # noqa: EM102, TRY003

    # Reset index and add row_id for traceability
    df = df.reset_index(drop=True).copy()
    df["row_id"] = df.index

    x = df.drop(columns=[target_col])
    y = df[target_col].astype(int)
    row_id = x.pop("row_id")

    # Initial development/test split
    x_dev, x_test, y_dev, y_test, id_dev, id_test = train_test_split(
        x, y, row_id,
        test_size=test_size,
        random_state=seed,
        stratify=y,
    )

    # Handle zeros in medical columns (0s --> NaNs)
    x_dev = _replace_zeros_with_nan(x_dev, MEDICAL_COLS)
    x_test = _replace_zeros_with_nan(x_test, MEDICAL_COLS)

    return {
        "x_dev": x_dev.reset_index(drop=True),
        "y_dev": y_dev.reset_index(drop=True),
        "x_test": x_test.reset_index(drop=True),
        "y_test": y_test.reset_index(drop=True),
        "row_id_dev": id_dev.reset_index(drop=True),
        "row_id_test": id_test.reset_index(drop=True),
        "artifacts": {
            "columns": list(x.columns),
            "seed": seed,
            "target_col": target_col,
            "row_id_dev": id_dev.reset_index(drop=True),
            "row_id_test": id_test.reset_index(drop=True),
        },
    }

def save_symbolic_data(
        x_dev: pd.DataFrame,
        y_dev: pd.Series,
        x_test: pd.DataFrame,
        y_test: pd.Series,
        seed: int = 42,
    ) -> None:
    """Save symbolic data for PyGol."""
    imputer = IterativeImputer(random_state=seed)

    x_dev_imputed = imputer.fit_transform(x_dev)
    x_test_imputed = imputer.transform(x_test)

    x_dev_symbolic = pd.DataFrame(x_dev_imputed, columns=x_dev.columns)
    x_test_symbolic = pd.DataFrame(x_test_imputed, columns=x_test.columns)

    dev_symbolic = pd.concat([x_dev_symbolic, y_dev.reset_index(drop=True)], axis=1)
    test_symbolic = pd.concat([x_test_symbolic, y_test.reset_index(drop=True)], axis=1)

    full_symbolic = pd.concat([dev_symbolic, test_symbolic], axis=0).reset_index(drop=True)

    full_symbolic.to_csv(SYMBOLIC_DIR / "symbolic_diabetes.csv", index=False)

    logger.info(f"Symbolic data saved to {SYMBOLIC_DIR / 'symbolic_diabetes.csv'}")

def save_cv_data(bundle: dict[str, Any], output_dir: str) -> None:
    """Save development/test data to specified output directory."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    bundle["x_dev"].to_csv(output_dir / "x_dev.csv", index=False)
    bundle["y_dev"].to_csv(output_dir / "y_dev.csv", index=False)
    bundle["x_test"].to_csv(output_dir / "x_test.csv", index=False)
    bundle["y_test"].to_csv(output_dir / "y_test.csv", index=False)

    with (output_dir / "split_artifacts.pkl").open("wb") as f:
        pickle.dump(bundle["artifacts"], f)

    logger.info(f"Development/test data saved to {output_dir}")
    logger.info(f"Split artifacts saved to {output_dir / 'split_artifacts.pkl'}")

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
        default="outcome",
        help="Target column name.",
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.2,
        help="Test set fraction.",
    )
    args = parser.parse_args()

    try:
        # Load
        data = load_data(args.input)

        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Analyse
        perform_correlation_analysis(data, output_path=output_dir / "correlation_heatmap.png")

        # Split into train/test sets
        bundle = split_train_test(
            data,
            target_col=args.target,
            test_size=args.test_size,
            seed=42,
        )

        save_symbolic_data(
            bundle["x_dev"],
            bundle["y_dev"],
            bundle["x_test"],
            bundle["y_test"],
            seed=42,
        )

        # Save train/test data for CV pipeline
        save_cv_data(bundle, output_dir)
    except Exception as e:
        logger.exception(f"An error occurred during preprocessing: {e}")
