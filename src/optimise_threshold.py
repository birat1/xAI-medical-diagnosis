"""Optimise thresholds for classification models to maximise F1-Score."""
import logging
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import precision_recall_curve

from models import MLP

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

def find_optimal_threshold(
        y_true: np.ndarray,
        y_probs: np.ndarray,
        model_name: str = "Model",
    ) -> tuple[float, np.ndarray, np.ndarray]:
    """Find and plot the threshold that maximises the F1-Score."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_probs)

    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)

    best_idx = np.argmax(f1_scores[:-1])
    best_threshold = thresholds[best_idx]
    best_f1 = f1_scores[best_idx]

    logger.info(f"\n--- {model_name} Optimisation ---")
    logger.info(f"Optimal Threshold: {best_threshold:.4f}")
    logger.info(f"Max F1-Score at this threshold: {best_f1:.4f}")

    return best_threshold, thresholds, f1_scores[:-1]

x_test = pd.read_csv("../data/processed/x_test.csv")
y_test = pd.read_csv("../data/processed/y_test.csv").to_numpy().ravel()

# Random Forest
with Path("../models/rf_model.pkl").open("rb") as f:
    rf_model = pickle.load(f)
rf_probs = rf_model.predict_proba(x_test)[:, 1]
rf_thresh, rf_ts, rf_f1s = find_optimal_threshold(y_test, rf_probs, "Random Forest")

# MLP
mlp = MLP(x_test.shape[1])
mlp.load_state_dict(torch.load("../models/mlp_model.pth"))
mlp.eval()
with torch.no_grad():
    logits = mlp(torch.tensor(x_test.values, dtype=torch.float32))
    mlp_probs = torch.sigmoid(logits).numpy().ravel()
mlp_thresh, mlp_ts, mlp_f1s = find_optimal_threshold(y_test, mlp_probs, "MLP")

# Visualisation
plt.figure(figsize=(10, 5))
plt.plot(rf_ts, rf_f1s, label=f"RF (Best Thresh: {rf_thresh:.2f})")
plt.plot(mlp_ts, mlp_f1s, label=f"MLP (Best Thresh: {mlp_thresh:.2f})")
plt.axvline(0.5, color="red", linestyle="--", label="Default 0.5")
plt.xlabel("Threshold")
plt.ylabel("F1-Score")
plt.title("Threshold Optimisation for F1-Score")
plt.legend()
plt.savefig("../models/threshold_optimisation.png")
logger.info("\nPlot saved to ../models/threshold_optimisation.png")
