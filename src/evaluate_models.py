"""Evaluate trained models on test set and save results."""
import json
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, classification_report, f1_score, precision_recall_fscore_support
import torch

from models import MLP

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path("../data/processed")
MODELS_DIR = Path("../models")

def load_test() -> tuple[pd.DataFrame, np.ndarray]:
    """Load test dataset."""
    x_test = pd.read_csv(DATA_DIR / "x_test.csv")
    y_test = pd.read_csv(DATA_DIR / "y_test.csv").to_numpy().ravel()
    logger.info("Test data loaded successfully.")
    return x_test, y_test

def load_thresholds() -> tuple[float, float]:
    """Load optimal thresholds for models."""
    with (MODELS_DIR / "thresholds.json").open("r") as f:
        data = json.load(f)
    return float(data["rf_threshold"]), float(data["mlp_threshold"])

def rf_probs(x: pd.DataFrame) -> np.ndarray:
    """Load trained Random Forest model and predict probabilities."""
    with (MODELS_DIR / "rf_model.pkl").open("rb") as f:
        rf_model = pickle.load(f)
    return rf_model.predict_proba(x)[:, 1]

def mlp_probs(x: pd.DataFrame) -> np.ndarray:
    """Load trained MLP model and predict probabilities."""
    mlp = MLP(x.shape[1])
    mlp.load_state_dict(torch.load(MODELS_DIR / "mlp_model.pth", weights_only=True))
    mlp.eval()
    with torch.no_grad():
        logits = mlp(torch.tensor(x.values, dtype=torch.float32))
        return torch.sigmoid(logits).numpy().ravel()

def get_metrics(name: str, y: np.ndarray, probs: np.ndarray, threshold: float) -> dict:
    """Calculate and log evaluation metrics."""
    preds = (probs >= threshold).astype(int)
    f1 = f1_score(y, preds)
    auprc = average_precision_score(y, probs)
    precision, recall, _, _ = precision_recall_fscore_support(y, preds, average="binary")

    logger.info(f"\n{name} Evaluation (threshold={threshold:.4f}) ---")
    logger.info(f"F1-Score: {f1:.4f} | AUPRC: {auprc:.4f}")
    logger.info(classification_report(y, preds))

    return {
        "f1_score": float(f1),
        "auprc": float(auprc),
        "precision": float(precision),
        "recall": float(recall),
        "threshold": float(threshold),
    }

if __name__ == "__main__":
    try:
        x_test, y_test = load_test()
        rf_threshold, mlp_threshold = load_thresholds()

        # Generate metrics for both models
        rf_results = get_metrics("Random Forest (Test)", y_test, rf_probs(x_test), rf_threshold)
        mlp_results = get_metrics("Multi-Layer Perceptron (Test)", y_test, mlp_probs(x_test), mlp_threshold)

        # Save results to JSON
        final_results = {
            "random_forest": rf_results,
            "mlp": mlp_results,
        }

        results_file = MODELS_DIR / "test_evaluation_results.json"
        with results_file.open("w") as f:
            json.dump(final_results, f, indent=4)

        logger.info(f"\nFinal test results saved to {results_file}")
    except Exception as e:
        logger.exception(f"An error occurred during evaluation: {e}")
