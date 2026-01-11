"""Evaluate trained models on Test dataset."""
import json
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, f1_score

from models import MLP

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path("../data/processed")
MODELS_DIR = Path("../models")

def load_test() -> tuple[pd.DataFrame, np.ndarray]:
    """Load preprocessed test dataset."""
    x_test = pd.read_csv(DATA_DIR / "x_test.csv")
    y_test = pd.read_csv(DATA_DIR / "y_test.csv").to_numpy().ravel()
    return x_test, y_test

def load_thresholds() -> tuple[float, float]:
    """Load optimal thresholds for RF and MLP from JSON file."""
    with (MODELS_DIR / "thresholds.json").open("r") as f:
        data = json.load(f)
    return float(data["rf_threshold"]), float(data["mlp_threshold"])

def rf_probs(x: pd.DataFrame) -> np.ndarray:
    """Load trained RF model and predict probabilities."""
    with (MODELS_DIR / "rf_model.pkl").open("rb") as f:
        rf_model = pickle.load(f)
    return rf_model.predict_proba(x)[:, 1]

def mlp_probs(x: pd.DataFrame) -> np.ndarray:
    """Load trained MLP model and predict probabilities."""
    model = MLP(x.shape[1])
    model.load_state_dict(torch.load(MODELS_DIR / "mlp_model.pth"))
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(x.values, dtype=torch.float32))
        return torch.sigmoid(logits).numpy().ravel()

def report(name: str, y: np.ndarray, probs: np.ndarray, threshold: float) -> None:
    """Generate and log classification report."""
    preds = (probs >= threshold).astype(int)
    logger.info(f"\n{name} (threshold={threshold:.4f})")
    logger.info(f"F1 Score: {f1_score(y, preds):.4f}")
    logger.info(classification_report(y, preds))

if __name__ == "__main__":
    x_test, y_test = load_test()
    rf_threshold, mlp_threshold = load_thresholds()

    report("Random Forest (Test)", y_test, rf_probs(x_test), rf_threshold)
    report("Multi-Layer Perceptron (Test)", y_test, mlp_probs(x_test), mlp_threshold)
