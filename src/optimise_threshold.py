"""Optimise thresholds for classification models to maximise F1-Score."""
import json
import logging
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.calibration import calibration_curve
from sklearn.metrics import average_precision_score, precision_recall_curve

from models import MLP

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path("../data/processed")
MODELS_DIR = Path("../models")
METRICS_DIR = Path("../results/metrics")

def find_optimal_threshold(
        y_true: np.ndarray,
        y_probs: np.ndarray,
        model_name: str = "Model",
    ) -> tuple[float, np.ndarray, np.ndarray, float]:
    """Find threshold that maximises the F1-Score."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_probs)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)

    best_idx = int(np.argmax(f1_scores[:-1]))
    best_threshold = float(thresholds[best_idx])
    best_f1 = float(f1_scores[best_idx])

    auprc = average_precision_score(y_true, y_probs)

    logger.info(f"\n--- {model_name} Optimisation (VALIDATION) ---")
    logger.info(f"AUPRC: {auprc:.4f}")
    logger.info(f"Optimal Threshold: {best_threshold:.4f}")
    logger.info(f"Max F1-Score at this threshold: {best_f1:.4f}")

    return best_threshold, thresholds, f1_scores[:-1], best_f1

if __name__ == "__main__":
    # Load Validation Set
    x_val = pd.read_csv(DATA_DIR / "x_val.csv")
    y_val = pd.read_csv(DATA_DIR / "y_val.csv").to_numpy().ravel()

    # 1. Random Forest Predictions
    with (MODELS_DIR / "rf_model.pkl").open("rb") as f:
            rf_model = pickle.load(f)
    rf_val_probs = rf_model.predict_proba(x_val)[:, 1]

    # 2. Decision Tree Predictions
    with (MODELS_DIR / "dt_model.pkl").open("rb") as f:
            dt_model = pickle.load(f)
    dt_val_probs = dt_model.predict_proba(x_val)[:, 1]

    # 3. Multi-Layer Perceptron Predictions
    mlp = MLP(x_val.shape[1])
    state = torch.load(MODELS_DIR / "mlp_model.pth", weights_only=True)
    mlp.load_state_dict(state)
    mlp.eval()
    with torch.no_grad():
        logits = mlp(torch.tensor(x_val.values, dtype=torch.float32))
    mlp_val_probs = torch.sigmoid(logits).numpy().ravel()

    # 3. Find Optimal Thresholds
    rf_thresh, rf_ts, rf_f1s, rf_best_f1 = find_optimal_threshold(y_val, rf_val_probs, model_name="Random Forest")
    dt_thresh, dt_ts, dt_f1s, dt_best_f1 = find_optimal_threshold(y_val, dt_val_probs, model_name="Decision Tree")
    mlp_thresh, mlp_ts, mlp_f1s, mlp_best_f1 = find_optimal_threshold(y_val, mlp_val_probs, model_name="Multi-Layer Perceptron")  # noqa: E501

    # 4. Save Artifacts
    thresholds = {
        "rf_threshold": rf_thresh,
        "dt_threshold": dt_thresh,
        "mlp_threshold": mlp_thresh,
        "rf_best_f1": rf_best_f1,
        "dt_best_f1": dt_best_f1,
        "mlp_best_f1": mlp_best_f1,
        "tuned_on": "validation",
    }
    with (METRICS_DIR / "thresholds.json").open("w") as f:
        json.dump(thresholds, f, indent=4)
    logger.info(f"\nOptimal thresholds saved to {METRICS_DIR / 'thresholds.json'}")

    # 5. Visualisation
    plt.figure(figsize=(12, 5))

    # Subplot 1: F1-Score vs Threshold
    plt.subplot(1, 2, 1)
    plt.plot(rf_ts, rf_f1s, label=f"RF (best={rf_thresh:.2f})")
    plt.plot(dt_ts, dt_f1s, label=f"DT (best={dt_thresh:.2f})")
    plt.plot(mlp_ts, mlp_f1s, label=f"MLP (best={mlp_thresh:.2f})")
    plt.axvline(0.5, color="red", linestyle="--", label="Default 0.5")
    plt.xlabel("Threshold")
    plt.ylabel("F1-Score")
    plt.title("Threshold Optimisation")
    plt.legend()

    # Subplot 2: Calibration Curve
    plt.subplot(1, 2, 2)
    for name, probs in [("RF", rf_val_probs), ("DT", dt_val_probs), ("MLP", mlp_val_probs)]:
        prob_true, prob_pred = calibration_curve(y_val, probs, n_bins=10)
        plt.plot(prob_pred, prob_true, marker="o", label=name)
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfectly Calibrated")
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Fraction of Positives")
    plt.title("Calibration Curve")
    plt.legend()

    plt.tight_layout()
    plt.savefig(METRICS_DIR / "optimisation_results.png")
    logger.info(f"\nPlot saved to {METRICS_DIR / 'optimisation_results.png'}")
