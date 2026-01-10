"""Data preprocessing module for cleaning and normalizing datasets."""
import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


def load_data(file_path: str) -> pd.DataFrame:
    """Load dataset from a CSV file."""
    if not Path(file_path).exists():
        raise FileNotFoundError(f"The file {file_path} was not found.")  # noqa: EM102, TRY003
    data = pd.read_csv(file_path)
    logger.info(f"Data loaded successfully from {file_path}")  # noqa: G004
    return data

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the dataset by handling missing values and outliers."""
    medical_cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    cleaned_df = df.copy()

    for col in medical_cols:
        if col in cleaned_df.columns:
            cleaned_df[col] = cleaned_df[col].replace(0, np.nan)

            median_val = cleaned_df[col].median()
            cleaned_df[col] = cleaned_df[col].fillna(median_val)

    logger.info("Data cleaning completed.")
    return cleaned_df

def perform_correlation_analysis(df: pd.DataFrame, output_path: str | None = None) -> None:
    """Perform correlation analysis to ensure high-quality input features."""
    plt.figure(figsize=(10, 8))
    correlation_matrix = df.corr()
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Feature Correlation Heatmap")

    if output_path:
        plt.savefig(output_path)
        logger.info(f"Correlation heatmap saved to {output_path}")  # noqa: G004
    else:
        plt.show()

def preprocess_and_split(
    df: pd.DataFrame,
    target_col: str = "Outcome",
    test_size: float = 0.2,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, StandardScaler]:
    """Split data into train and test sets. Scale features using StandardScaler."""
    x = df.drop(columns=[target_col])
    y = df[target_col]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=42, stratify=y)

    scaler = StandardScaler()
    x_train_scaled = pd.DataFrame(scaler.fit_transform(x_train), columns=x_train.columns)
    x_test_scaled = pd.DataFrame(scaler.transform(x_test), columns=x_test.columns)

    logger.info(f"Completed data splitting and scaling. Train size: {len(x_train)}, Test size: {len(x_test)}")  # noqa: G004
    return x_train_scaled, x_test_scaled, y_train, y_test, scaler

def save_processed_data(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    output_dir: str,
    ) -> None:
    """Save processed data to specified output directory."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


    x_train.to_csv(output_dir / "x_train.csv", index=False)
    x_test.to_csv(output_dir / "x_test.csv", index=False)
    y_train.to_csv(output_dir / "y_train.csv", index=False)
    y_test.to_csv(output_dir / "y_test.csv", index=False)
    logger.info(f"Processed data saved to {output_dir}")  # noqa: G004

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
        help="Directory to save the processed data.",
    )
    args = parser.parse_args()

    try:
        # Load and Clean
        data = load_data(args.input)
        cleaned_data = clean_data(data)

        output_dir = args.output if args.output.endswith("/") else args.output + "/"
        # Analyse
        perform_correlation_analysis(cleaned_data, output_path=output_dir + "correlation_heatmap.png")

        # Preprocess
        x_train, x_test, y_train, y_test, scaler = preprocess_and_split(cleaned_data)

        # Save
        save_processed_data(x_train, x_test, y_train, y_test, output_dir)
    except Exception as e:
        logger.exception(f"An error occurred during preprocessing: {e}") #noqa: G004
