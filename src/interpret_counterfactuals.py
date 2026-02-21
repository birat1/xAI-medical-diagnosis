"""Convert DiCE counterfactuals into an understandable format for clinicians."""
import json
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

ARTIFACTS_DIR = Path("../data/processed")
DICE_DIR = Path("../results/dice")
OUTPUT_DIR = Path("../results/interpretations")
RAW_DATA_DIR = Path("../data/raw")

def load_artifacts() -> dict:
    """Load preprocessing artifacts needed to interpret counterfactuals."""
    artifacts_path = ARTIFACTS_DIR / "preprocess_artifacts.pkl"

    if not artifacts_path.exists():
        raise FileNotFoundError(f"Preprocessing artifacts not found at {artifacts_path}")  # noqa: EM102, TRY003

    with artifacts_path.open("rb") as f:
        artifacts = pickle.load(f)
        logger.info(f"Preprocessing artifacts loaded from {artifacts_path}\n")
    return artifacts

def process_patient_cfs(
        json_path: Path,
        scaler: StandardScaler,
        target_cols: list[str],
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict, int]:
    """Interpret counterfacts for a patient file."""
    with json_path.open("r") as f:
        dice_results = json.load(f)

    feature_names = dice_results["feature_names"]
    metadata = dice_results.get("metadata", {})
    patient_idx = int(metadata.get("patient_idx", 0))

    # Map feature names to target columns to ensure correct ordering
    idx_map = [feature_names.index(col) for col in target_cols]

    # Extract data and the outcome
    test_data_row = dice_results["test_data"][0][0]
    cfs_list = dice_results["cfs_list"][0]

    # Original Prediction and CF predictions
    imp_outcomes = test_data_row[-1]
    cf_outcomes = [cf[-1] for cf in cfs_list]

    # Extract features
    x_scaled = np.array(test_data_row, dtype=float)[idx_map]
    cfs_scaled = np.array([cf[:len(feature_names)] for cf in cfs_list], dtype=float)[:, idx_map]

    # Inverse transform to original clinical units
    x_imp = scaler.inverse_transform(x_scaled.reshape(1, -1)).ravel()
    cfs_imp = scaler.inverse_transform(cfs_scaled)

    # Create DataFrames for original and counterfactuals
    df_imp = pd.DataFrame([x_imp], columns=target_cols, index=["imp"])
    df_imp["Outcome"] = imp_outcomes
    df_cfs = pd.DataFrame(cfs_imp, columns=target_cols, index=[f"cf_{i+1}" for i in range(len(cfs_imp))])
    df_cfs["Outcome"] = cf_outcomes

    # Calculate the delta (change needed) for each feature
    df_delta = df_cfs.drop(columns="Outcome") - df_imp.drop(columns="Outcome").iloc[0]

    return df_imp, df_cfs, df_delta, metadata, patient_idx

if __name__ == "__main__":
    try:
        artifacts = load_artifacts()
        scaler = artifacts["scaler"]
        target_cols = artifacts["columns"]
        row_id_test = artifacts.get("row_id_test")
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        raw_df = pd.read_csv(RAW_DATA_DIR / "diabetes.csv").reset_index(drop=True)

        json_files = list(DICE_DIR.rglob("dice_counterfactuals.json"))
        all_summaries = []

        for json_path in json_files:
            model_name = json_path.parent.parent.name
            patient_folder = json_path.parent.name

            logger.info(f"Processing MODEL: {model_name} | Patient Folder: {patient_folder}")

            df_imp, df_cfs, df_delta, metadata, patient_idx = process_patient_cfs(json_path, scaler, target_cols)

            raw_row_id = None
            raw_row = None
            if row_id_test is not None:
                raw_row_id = int(row_id_test.iloc[patient_idx])
                raw_row = raw_df.iloc[raw_row_id]
            else:
                logger.warning("row_id_test not found in artifacts. Cannot map DiCE patient_idx to raw diabetes.csv row.")

            # Save summary of significant changes for clinicians
            with (OUTPUT_DIR / f"{model_name}_{patient_folder}_metadata.json").open("w") as f:
                f.write("DiCE Counterfactual Interpretation Summary\n")
                f.write(f"{'-' * 75}\n")
                f.write(f"Model: {model_name}\n")
                f.write(f"Patient Folder: {patient_folder}\n")
                f.write(f"Patient Index (Test Set): {patient_idx}\n")

                if raw_row_id is not None:
                    f.write(f"Row (diabetes.csv): {raw_row_id + 2}\n\n")

                    raw_features = raw_row[target_cols].to_frame().T
                    raw_features.index = ["raw"]

                    f.write("RAW Patient Data (Clinical Units - may include 0=missing):\n")
                    f.write(raw_features.round(3).to_string())
                    f.write("\n\n")

                f.write("IMPUTED Patient Data (Clinical Units):\n")
                f.write(df_imp.round(3).to_string())
                f.write("\n\n")

                f.write("Counterfactuals (Clinical Units):\n")
                f.write(df_cfs.round(3).to_string())
                f.write("\n\n")

                f.write("Required Changes (Δ = CF - IMP):\n")
                f.write(df_delta.round(3).to_string())
                f.write("\n\n")

                # Analyze which features change most frequently across counterfactuals
                abs_changes = df_delta.abs()
                change_frequency = (abs_changes > 1e-6).sum().sort_values(ascending=False)

                f.write("Feature Change Frequency Across Counterfactuals:\n")
                f.write(change_frequency.to_string())
                f.write("\n")

            logger.info(f"Saved interpretation files for MODEL: {model_name} | Patient Folder: {patient_folder}\n")

        logger.info(f"All counterfactual interpretations saved to {OUTPUT_DIR}")
    except Exception as e:
        logger.exception(f"An error occurred: {e}")  # noqa: TRY401

