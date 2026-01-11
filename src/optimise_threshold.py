"""Optimise thresholds for classification models to maximise F1-Score."""
import json
import logging
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, f1_score, precision_recall_curve

from models import MLP

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path("../data/processed")
MODELS_DIR = Path("../models")

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

    logger.info(f"\n--- {model_name} Optimisation (VALIDATION) ---")
    logger.info(f"Optimal Threshold: {best_threshold:.4f}")
    logger.info(f"Max F1-Score at this threshold: {best_f1:.4f}")

    return best_threshold, thresholds, f1_scores[:-1], best_f1

# Random Forest
def predict_rf_probs(x: pd.DataFrame) -> np.ndarray:
    """Predict probabilities using the saved Random Forest model."""
    with (MODELS_DIR / "rf_model.pkl").open("rb") as f:
        rf_model = pickle.load(f)
    return rf_model.predict_proba(x)[:, 1]

# MLP
def predict_mlp_probs(x: pd.DataFrame) -> np.ndarray:
    """Predict probabilities using the saved MLP model."""
    mlp = MLP(x.shape[1])
    state = torch.load(MODELS_DIR / "mlp_model.pth", weights_only=True)
    mlp.load_state_dict(state)
    mlp.eval()
    with torch.no_grad():
        logits = mlp(torch.tensor(x.values, dtype=torch.float32))
        return torch.sigmoid(logits).numpy().ravel()

# Evaluation
def evaluate_threshold(y_true: np.ndarray, probs: np.ndarray, threshold: float, name: str) -> None:
    """Evaluate model performance at a given threshold."""
    preds = (probs >= threshold).astype(int)
    logger.info(f"\n--- {name} Evaluation (threshold {threshold:.4f}) ---")
    logger.info(f"F1-Score: {f1_score(y_true, preds):.4f}")
    logger.info(classification_report(y_true, preds))

if __name__ == "__main__":
    # Load Validation Set
    x_val_path = DATA_DIR / "x_val.csv"
    y_val_path = DATA_DIR / "y_val.csv"
    if not x_val_path.exists() or not y_val_path.exists():
        raise FileNotFoundError("Validation data not found. Please run preprocessing first.")  # noqa: EM101, TRY003

    x_val = pd.read_csv(x_val_path)
    y_val = pd.read_csv(y_val_path).to_numpy().ravel()

    # Tune thresholds
    rf_val_probs = predict_rf_probs(x_val)
    rf_thresh, rf_ts, rf_f1s, rf_best_f1 = find_optimal_threshold(y_val, rf_val_probs, model_name="Random Forest")

    mlp_val_probs = predict_mlp_probs(x_val)
    mlp_thresh, mlp_ts, mlp_f1s, mlp_best_f1 = find_optimal_threshold(y_val, mlp_val_probs, model_name="Multi-Layer Perceptron")

    # Save thresholds
    thresholds = {
        "rf_threshold": rf_thresh,
        "mlp_threshold": mlp_thresh,
        "rf_best_f1": rf_best_f1,
        "mlp_best_f1": mlp_best_f1,
        "tuned_on": "validation",
    }
    with (MODELS_DIR / "thresholds.json").open("w") as f:
        json.dump(thresholds, f, indent=4)
    logger.info(f"\nOptimal thresholds saved to {MODELS_DIR / 'thresholds.json'}")

    # Visualisation
    plt.figure(figsize=(10, 5))
    plt.plot(rf_ts, rf_f1s, label=f"RF (best={rf_thresh:.2f})")
    plt.plot(mlp_ts, mlp_f1s, label=f"MLP (best={mlp_thresh:.2f})")
    plt.axvline(0.5, color="red", linestyle="--", label="Default 0.5")
    plt.xlabel("Threshold")
    plt.ylabel("F1-Score")
    plt.title("Threshold Optimisation on Validation Set")
    plt.legend()
    plt.savefig(MODELS_DIR / "threshold_optimisation_val.png", bbox_inches="tight")
    logger.info(f"\nPlot saved to {MODELS_DIR / 'threshold_optimisation_val.png'}")

    # Evaluate on Test Set
    x_test = pd.read_csv(DATA_DIR / "x_test.csv")
    y_test = pd.read_csv(DATA_DIR / "y_test.csv").to_numpy().ravel()

    evaluate_threshold(y_test, predict_rf_probs(x_test), rf_thresh, "Random Forest (Test)")
    evaluate_threshold(y_test, predict_mlp_probs(x_test), mlp_thresh, "Multi-Layer Perceptron (Test)")
