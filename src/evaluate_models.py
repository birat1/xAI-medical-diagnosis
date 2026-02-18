"""Evaluate trained models on test set and save results."""
import json
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, classification_report, f1_score, precision_recall_fscore_support

from models import MLP

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path("../data/processed")
MODELS_DIR = Path("../models")
METRICS_DIR = Path("../results/metrics")

def load_test() -> tuple[pd.DataFrame, np.ndarray]:
    """Load test dataset."""
    x_test = pd.read_csv(DATA_DIR / "x_test.csv")
    y_test = pd.read_csv(DATA_DIR / "y_test.csv").to_numpy().ravel()
    logger.info("Test data loaded successfully.")
    return x_test, y_test

def load_thresholds() -> tuple[float, float, float]:
    """Load optimal thresholds for models."""
    with (METRICS_DIR / "thresholds.json").open("r") as f:
        data = json.load(f)
    return float(data["rf_threshold"]), float(data["dt_threshold"]), float(data["mlp_threshold"])

def get_probs(model_name: str, x: pd.DataFrame) -> np.ndarray:
    """Get predicted probabilities for a given model."""
    model_path = MODELS_DIR / model_name

    if model_path.suffix == ".pkl":
        with model_path.open("rb") as f:
            model = pickle.load(f)
        return model.predict_proba(x)[:, 1]

    if model_path.suffix == ".pth":
        mlp = MLP(x.shape[1])
        mlp.load_state_dict(torch.load(model_path, weights_only=True))
        mlp.eval()
        with torch.no_grad():
            logits = mlp(torch.tensor(x.values, dtype=torch.float32))
            return torch.sigmoid(logits).numpy().ravel()

    raise ValueError(f"Unsupported model format: {model_path.suffix}")  # noqa: EM102, TRY003


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
        rf_threshold, dt_threshold, mlp_threshold = load_thresholds()

        # Generate metrics for all models
        rf_results = get_metrics(
            "Random Forest (Test)",
            y_test,
            get_probs("rf_model.pkl", x_test),
            rf_threshold,
        )
        dt_results = get_metrics(
            "Decision Tree (Test)",
            y_test,
            get_probs("dt_model.pkl", x_test),
            dt_threshold,
        )
        mlp_results = get_metrics(
            "Multi-Layer Perceptron (Test)",
            y_test,
            get_probs("mlp_model.pth", x_test),
            mlp_threshold,
        )

        # Save results to JSON
        final_results = {
            "random_forest": rf_results,
            "decision_tree": dt_results,
            "mlp": mlp_results,
        }

        results_file = METRICS_DIR / "test_evaluation_results.json"
        with results_file.open("w") as f:
            json.dump(final_results, f, indent=4)

        logger.info(f"\nFinal test results saved to {results_file}")
    except Exception as e:
        logger.exception(f"An error occurred during evaluation: {e}")
