"""Evaluate models on test set."""
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

def load_thresholds() -> dict[str, float]:
    """Load optimal thresholds for models."""
    with (METRICS_DIR / "thresholds.json").open("r") as f:
        data = json.load(f)

    return {
        "decision_tree": float(data["dt_threshold"]),
        "random_forest": float(data["rf_threshold"]),
        "mlp": float(data["mlp_threshold"]),
    }

def load_mlp(x: pd.DataFrame) -> MLP:
    """Load trained MLP model."""
    with (METRICS_DIR / "hyperparameters.json").open("r") as f:
        best_params = json.load(f)

    mlp_params = best_params["mlp"]
    hidden_layers = [
        mlp_params[f"hidden_size_{i}"]
        for i in range(mlp_params["n_layers"])
    ]

    model = MLP(
        input_size=x.shape[1],
        hidden_layers=hidden_layers,
        dropout=mlp_params["dropout"],
    )

    model.load_state_dict(torch.load(MODELS_DIR / "mlp_model.pth", weights_only=True))
    model.eval()

    return model

def get_metrics(name: str, y_true: np.ndarray, probs: np.ndarray, threshold: float) -> dict:
    """Calculate and log evaluation metrics."""
    preds = (probs >= threshold).astype(int)

    f1 = f1_score(y_true, preds)
    auprc = average_precision_score(y_true, probs)
    precision, recall, _, _ = precision_recall_fscore_support(y_true, preds, average="binary", zero_division=0)

    logger.info(f"\n{name} Test Evaluation")
    logger.info(f"Threshold: {threshold:.4f}")
    logger.info(f"F1: {f1:.4f} | AUPRC: {auprc:.4f}")
    logger.info(classification_report(y_true, preds, zero_division=0))

    return {
        "threshold": float(threshold),
        "f1_score": float(f1),
        "auprc": float(auprc),
        "precision": float(precision),
        "recall": float(recall),
    }

if __name__ == "__main__":
    x_test, y_test = load_test()
    thresholds = load_thresholds()

    with (MODELS_DIR / "dt_model.pkl").open("rb") as f:
        dt_model = pickle.load(f)

    with (MODELS_DIR / "rf_model.pkl").open("rb") as f:
        rf_model = pickle.load(f)

    mlp_model = load_mlp(x_test)

    dt_probs = dt_model.predict_proba(x_test)[:, 1]
    rf_probs = rf_model.predict_proba(x_test)[:, 1]

    with torch.no_grad():
        logits = mlp_model(torch.tensor(x_test.values, dtype=torch.float32))
        mlp_probs = torch.sigmoid(logits).cpu().numpy().ravel()

    # Save results to JSON
    final_results = {
        "decision_tree": {
            "default": get_metrics(
                "Decision Tree (default)",
                y_test,
                dt_probs,
                0.5,
            ),
            "tuned": get_metrics(
                "Decision Tree (tuned)",
                y_test,
                dt_probs,
                thresholds["decision_tree"],
            ),
        },
        "random_forest": {
            "default": get_metrics(
                "Random Forest (default)",
                y_test,
                rf_probs,
                0.5,
            ),
            "tuned": get_metrics(
                "Random Forest (tuned)",
                y_test,
                rf_probs,
                thresholds["random_forest"],
            ),
        },
        "mlp": {
            "default": get_metrics(
                "MLP (default)",
                y_test,
                mlp_probs,
                0.5,
            ),
            "tuned": get_metrics(
                "MLP (tuned)",
                y_test,
                mlp_probs,
                thresholds["mlp"],
            ),
        },
    }

    with (METRICS_DIR / "test_evaluation_results.json").open("w") as f:
        json.dump(final_results, f, indent=2)

    logger.info(f"\nFinal test results saved to {METRICS_DIR / 'test_evaluation_results.json'}")
