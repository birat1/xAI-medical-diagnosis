"""Optimise thresholds using validation set."""
import json
import logging
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.base import clone
from sklearn.calibration import calibration_curve
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.model_selection import StratifiedKFold
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

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
    ) -> tuple[float, np.ndarray, np.ndarray, float, float]:
    """Find threshold that maximises the F1-Score."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_probs)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)

    best_idx = int(np.argmax(f1_scores[:-1]))
    best_threshold = float(thresholds[best_idx])
    best_f1 = float(f1_scores[best_idx])
    auprc = average_precision_score(y_true, y_probs)

    logger.info(f"\n--- {model_name} Threshold Optimisation ---")
    logger.info(f"Validation AUPRC: {auprc:.4f}")
    logger.info(f"Best Threshold: {best_threshold:.4f}")
    logger.info(f"Best validation F1: {best_f1:.4f}")

    return {
        "threshold": best_threshold,
        "f1": best_f1,
        "auprc": auprc,
        "thresholds": thresholds,
        "f1_scores": f1_scores[:-1],
    }

def get_sklearn_oof_probs(
        model: object,
        x: pd.DataFrame,
        y: np.ndarray,
        model_name: str,
        n_splits: int = 5,
    ) -> np.ndarray:
    """Get out-of-fold predicted probabilities for sklearn model using Stratified K-Fold CV."""
    skf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=42,
    )

    oof_probs = np.zeros(len(y))

    for fold, (train_idx, val_idx) in enumerate(skf.split(x, y), start=1):
        logger.info(f"{model_name} Fold {fold}/{n_splits}")

        x_train, x_val = x.iloc[train_idx], x.iloc[val_idx]
        y_train = y[train_idx]

        fold_model = clone(model)
        fold_model.fit(x_train, y_train)

        oof_probs[val_idx] = fold_model.predict_proba(x_val)[:, 1]

    return oof_probs

def train_mlp_fold(  # noqa: PLR0913
        x_train: pd.DataFrame,
        y_train: np.ndarray,
        input_size: int,
        hidden_layers: list[int],
        dropout: float,
        lr: float,
        weight_decay: float,
        batch_size: int,
        epochs: int,
    ) -> MLP:
    """Train MLP for one fold of Stratified K-Fold CV."""
    model = MLP(
        input_size=input_size,
        hidden_layers=hidden_layers,
        dropout=dropout,
    )

    x_tensor = torch.tensor(x_train.values, dtype=torch.float32)
    y_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)

    dataset = TensorDataset(x_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    num_pos = np.sum(y_train == 1)
    num_neg = np.sum(y_train == 0)

    pos_weight = torch.tensor([num_neg / num_pos], dtype=torch.float32)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimiser = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    model.train()

    for _ in range(epochs):
        for xb, yb in loader:
            optimiser.zero_grad()

            logits = model(xb)
            loss = criterion(logits, yb)

            loss.backward()
            optimiser.step()

    return model

def get_mlp_oof_probs(  # noqa: PLR0913
        x: pd.DataFrame,
        y: np.ndarray,
        hidden_layers: list[int],
        dropout: float,
        lr: float,
        weight_decay: float,
        batch_size: int,
        epochs: int,
        n_splits: int = 5,
    ) -> np.ndarray:
    """Get out-of-fold predicted probabilities for MLP using Stratified K-Fold CV."""
    skf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=42,
    )

    oof_probs = np.zeros(len(y))

    for fold, (train_idx, val_idx) in enumerate(skf.split(x, y), start=1):
        logger.info(f"MLP Fold {fold}/{n_splits}")

        x_fold_train, y_fold_train = x.iloc[train_idx], y[train_idx]
        x_fold_val = x.iloc[val_idx]

        model = train_mlp_fold(
            x_train=x_fold_train,
            y_train=y_fold_train,
            input_size=x.shape[1],
            hidden_layers=hidden_layers,
            dropout=dropout,
            lr=lr,
            weight_decay=weight_decay,
            batch_size=batch_size,
            epochs=epochs,
        )

        model.eval()

        with torch.no_grad():
            logits = model(torch.tensor(x_fold_val.values, dtype=torch.float32))
            probs = torch.sigmoid(logits).cpu().numpy().ravel()

        oof_probs[val_idx] = probs

    return oof_probs


def load_mlp_hyperparameters() -> dict[str, object]:
    """Load best MLP hyperparameters."""
    with (METRICS_DIR / "hyperparameters.json").open("r") as f:
        best_params = json.load(f)

    mlp_params = best_params["mlp"]
    hidden_layers = [
        mlp_params[f"hidden_size_{i}"]
        for i in range(mlp_params["n_layers"])
    ]

    return {
        "hidden_layers": hidden_layers,
        "dropout": mlp_params["dropout"],
        "lr": mlp_params["lr"],
        "weight_decay": mlp_params["weight_decay"],
        "batch_size": mlp_params["batch_size"],
    }

if __name__ == "__main__":
    # Load Validation Set
    x_train = pd.read_csv(DATA_DIR / "x_train.csv")
    y_train = pd.read_csv(DATA_DIR / "y_train.csv").to_numpy().ravel()


    with (MODELS_DIR / "dt_model.pkl").open("rb") as f:
        dt_model = pickle.load(f)

    with (MODELS_DIR / "rf_model.pkl").open("rb") as f:
        rf_model = pickle.load(f)

    mlp_params = load_mlp_hyperparameters()

    dt_oof_probs = get_sklearn_oof_probs(dt_model, x_train, y_train, model_name="Decision Tree")
    rf_oof_probs = get_sklearn_oof_probs(rf_model, x_train, y_train, model_name="Random Forest")
    mlp_oof_probs = get_mlp_oof_probs(
        x=x_train,
        y=y_train,
        hidden_layers=mlp_params["hidden_layers"],
        dropout=mlp_params["dropout"],
        lr=mlp_params["lr"],
        weight_decay=mlp_params["weight_decay"],
        batch_size=mlp_params["batch_size"],
        epochs=150,
    )

    # 3. Find Optimal Thresholds
    dt_results = find_optimal_threshold(y_train, dt_oof_probs, model_name="Decision Tree")
    rf_results = find_optimal_threshold(y_train, rf_oof_probs, model_name="Random Forest")
    mlp_results = find_optimal_threshold(y_train, mlp_oof_probs, model_name="Multi-Layer Perceptron")

    # 4. Save Artifacts
    thresholds = {
        "dt_threshold": dt_results["threshold"],
        "rf_threshold": rf_results["threshold"],
        "mlp_threshold": mlp_results["threshold"],

        "dt_cv_f1": dt_results["f1"],
        "rf_cv_f1": rf_results["f1"],
        "mlp_cv_f1": mlp_results["f1"],

        "dt_cv_auprc": dt_results["auprc"],
        "rf_cv_auprc": rf_results["auprc"],
        "mlp_cv_auprc": mlp_results["auprc"],

        "tuned_on": "cv_oof_probs",
    }

    with (METRICS_DIR / "thresholds.json").open("w") as f:
        json.dump(thresholds, f, indent=2)
    logger.info(f"\nCV thresholds saved to {METRICS_DIR / 'thresholds.json'}")

    # 5. Visualisation
    plt.figure(figsize=(12, 5))

    # Subplot 1: F1-Score vs Threshold
    plt.subplot(1, 2, 1)
    plt.plot(dt_results["thresholds"], dt_results["f1_scores"], label=f"DT (best={dt_results['threshold']:.2f})")
    plt.plot(rf_results["thresholds"], rf_results["f1_scores"], label=f"RF (best={rf_results['threshold']:.2f})")
    plt.plot(mlp_results["thresholds"], mlp_results["f1_scores"], label=f"MLP (best={mlp_results['threshold']:.2f})")
    plt.axvline(0.5, color="red", linestyle="--", label="Default 0.5")
    plt.xlabel("Threshold")
    plt.ylabel("F1-Score")
    plt.title("CV F1-Score vs Threshold")
    plt.legend()

    # Subplot 2: Calibration Curve
    plt.subplot(1, 2, 2)

    for name, probs in [
        ("DT", dt_oof_probs),
        ("RF", rf_oof_probs),
        ("MLP", mlp_oof_probs),
    ]:
        prob_true, prob_pred = calibration_curve(y_train, probs, n_bins=10)
        plt.plot(prob_pred, prob_true, marker="o", label=name)
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfectly Calibrated")
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Fraction of Positives")
    plt.title("OOF Calibration Curve")
    plt.legend()

    plt.tight_layout()
    plt.savefig(METRICS_DIR / "optimisation_results.png")
    logger.info(f"\nPlot saved to {METRICS_DIR / 'optimisation_results.png'}")
