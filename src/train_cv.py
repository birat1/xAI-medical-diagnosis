"""5-fold cross-validation for training."""
import json
import logging
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

from models import MLP

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path("../data/processed")
MODELS_DIR = Path("../models")
MODELS_DIR.mkdir(exist_ok=True)
METRICS_DIR = Path("../results/metrics")
METRICS_DIR.mkdir(exist_ok=True)
RESULTS_DIR = Path("../results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed: int = 42) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.default_rng(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_data() -> tuple[pd.DataFrame, np.ndarray]:
    """Load development data for 5-fold CV."""
    x_dev = pd.read_csv(DATA_DIR / "x_dev.csv")
    y_dev = pd.read_csv(DATA_DIR / "y_dev.csv").to_numpy().ravel()

    return x_dev, y_dev

def fit_preprocessor(x_train_fold: pd.DataFrame, seed: int = 42) -> tuple[IterativeImputer, StandardScaler, np.ndarray]:
    """Fit imputer and scaler on training fold."""
    imputer = IterativeImputer(random_state=seed)
    scaler = StandardScaler()

    x_train_imputed = imputer.fit_transform(x_train_fold)
    x_train_scaled = scaler.fit_transform(x_train_imputed)

    return imputer, scaler, x_train_scaled

def transform_fold(x_fold: pd.DataFrame, imputer: IterativeImputer, scaler: StandardScaler) -> np.ndarray:
    """Transform fold using fitted preprocessor."""
    x_imputed = imputer.transform(x_fold)
    x_scaled = scaler.transform(x_imputed)

    return x_scaled

def find_best_threshold(y_true: np.ndarray, y_probs: np.ndarray) -> tuple[float, float, float, float]:
    """Find the best threshold that maximises F1 score for one fold."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_probs)

    precision = precision[:-1]
    recall = recall[:-1]

    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
    best_idx = int(np.argmax(f1_scores))

    return (
        float(thresholds[best_idx]),
        float(f1_scores[best_idx]),
        float(precision[best_idx]),
        float(recall[best_idx]),
    )

###
###
###

def run_dt_cv(x_dev: pd.DataFrame, y_dev: np.ndarray, n_splits: int = 5, seed: int = 42) -> dict[str, object]:
    """Run 5-fold CV for Decision Tree."""
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    fold_auprc: list[float] = []
    fold_f1: list[float] = []
    fold_precision: list[float] = []
    fold_recall: list[float] = []
    fold_threshold: list[float] = []

    oof_probs = np.zeros(len(y_dev), dtype=float)

    for fold, (train_idx, val_idx) in enumerate(cv.split(x_dev, y_dev), start=1):
        logger.info(f"[DT] Fold {fold}/{n_splits}")

        x_train_fold, y_train_fold = x_dev.iloc[train_idx].copy(), y_dev[train_idx]
        x_val_fold, y_val_fold = x_dev.iloc[val_idx].copy(), y_dev[val_idx]

        imputer, scaler, x_train_fold_scaled = fit_preprocessor(x_train_fold, seed)
        x_val_fold_scaled = transform_fold(x_val_fold, imputer, scaler)

        model = DecisionTreeClassifier(
            max_depth=10,
            min_samples_split=2,
            min_samples_leaf=1,
            criterion="entropy",
            ccp_alpha=0.0005,
            class_weight="balanced",
            random_state=42,
        )
        model.fit(x_train_fold_scaled, y_train_fold)

        val_probs = model.predict_proba(x_val_fold_scaled)[:, 1]
        oof_probs[val_idx] = val_probs

        auprc = average_precision_score(y_val_fold, val_probs)
        best_threshold, best_f1, best_precision, best_recall = find_best_threshold(y_val_fold, val_probs)

        fold_auprc.append(float(auprc))
        fold_f1.append(float(best_f1))
        fold_precision.append(float(best_precision))
        fold_recall.append(float(best_recall))
        fold_threshold.append(float(best_threshold))

        logger.info(
            f"[DT] Fold {fold} | "
            f"AUPRC: {auprc:.4f} | "
            f"F1: {best_f1:.4f} | "
            f"Precision: {best_precision:.4f} | "
            f"Recall: {best_recall:.4f} | "
            f"Threshold: {best_threshold:.4f}",
        )

    return {
        "fold_auprc": fold_auprc,
        "fold_f1": fold_f1,
        "fold_precision": fold_precision,
        "fold_recall": fold_recall,
        "fold_threshold": fold_threshold,
        "mean_auprc": float(np.mean(fold_auprc)),
        "mean_f1": float(np.mean(fold_f1)),
        "mean_precision": float(np.mean(fold_precision)),
        "mean_recall": float(np.mean(fold_recall)),
        "mean_threshold": float(np.mean(fold_threshold)),
        "oof_probs": oof_probs,
    }

def run_rf_cv(x_dev: pd.DataFrame, y_dev: np.ndarray, n_splits: int = 5, seed: int = 42) -> dict[str, object]:
    """Run 5-fold CV for Random Forest."""
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    fold_auprc: list[float] = []
    fold_f1: list[float] = []
    fold_precision: list[float] = []
    fold_recall: list[float] = []
    fold_threshold: list[float] = []

    oof_probs = np.zeros(len(y_dev), dtype=float)

    for fold, (train_idx, val_idx) in enumerate(cv.split(x_dev, y_dev), start=1):
        logger.info(f"[RF] Fold {fold}/{n_splits}")

        x_train_fold, y_train_fold = x_dev.iloc[train_idx].copy(), y_dev[train_idx]
        x_val_fold, y_val_fold = x_dev.iloc[val_idx].copy(), y_dev[val_idx]

        imputer, scaler, x_train_fold_scaled = fit_preprocessor(x_train_fold, seed)
        x_val_fold_scaled = transform_fold(x_val_fold, imputer, scaler)

        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_leaf=1,
            min_samples_split=2,
            class_weight="balanced",
            max_features="log2",
            random_state=42,
        )
        model.fit(x_train_fold_scaled, y_train_fold)

        val_probs = model.predict_proba(x_val_fold_scaled)[:, 1]
        oof_probs[val_idx] = val_probs

        auprc = average_precision_score(y_val_fold, val_probs)
        best_threshold, best_f1, best_precision, best_recall = find_best_threshold(y_val_fold, val_probs)

        fold_auprc.append(float(auprc))
        fold_f1.append(float(best_f1))
        fold_precision.append(float(best_precision))
        fold_recall.append(float(best_recall))
        fold_threshold.append(float(best_threshold))

        logger.info(
            f"[RF] Fold {fold} | "
            f"AUPRC: {auprc:.4f} | "
            f"F1: {best_f1:.4f} | "
            f"Precision: {best_precision:.4f} | "
            f"Recall: {best_recall:.4f} | "
            f"Threshold: {best_threshold:.4f}",
        )

    return {
        "fold_auprc": fold_auprc,
        "fold_f1": fold_f1,
        "fold_precision": fold_precision,
        "fold_recall": fold_recall,
        "fold_threshold": fold_threshold,
        "mean_auprc": float(np.mean(fold_auprc)),
        "mean_f1": float(np.mean(fold_f1)),
        "mean_precision": float(np.mean(fold_precision)),
        "mean_recall": float(np.mean(fold_recall)),
        "mean_threshold": float(np.mean(fold_threshold)),
        "oof_probs": oof_probs,
    }

def train_mlp_fold(  # noqa: PLR0913, PLR0915
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int = 500,
        lr: float = 1e-3,
        seed: int = 42,
    ) -> tuple[np.ndarray, float]:
    """Train one MLP fold and return validation probabilities and AUPRC."""
    set_seed(seed)

    model = MLP(x_train.shape[1]).to(device)

    # 1. Weighted Loss: handle class imbalance
    pos = float((y_train == 1).sum())
    neg = float((y_train == 0).sum())
    pos_weight = torch.tensor([neg / pos], dtype=torch.float32, device=device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimiser = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-3)

    # 2. LR Scheduler: reduces learning rate on plateau of validation loss
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimiser, mode="min", factor=0.5, patience=6, min_lr=1e-6)

    # Convert data to tensors
    x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)

    x_val_tensor = torch.tensor(x_val, dtype=torch.float32, device=device)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32, device=device).view(-1, 1)

    train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, drop_last=False)

    best_state = None
    best_val_auprc = -float("inf")
    early_stop_patience = 20
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        # Training phase
        model.train()
        train_loss = 0.0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimiser.zero_grad(set_to_none=True)
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimiser.step()
            train_loss += loss.item() * batch_x.size(0)

        train_loss /= len(train_loader.dataset)

        # 3. Validation phase for early stopping and LR scheduling
        model.eval()
        with torch.no_grad():
            val_logits = model(x_val_tensor)
            val_loss = criterion(val_logits, y_val_tensor)

            val_probs = torch.sigmoid(val_logits).cpu().numpy().ravel()
            val_auprc = average_precision_score(y_val, val_probs)

        # Step the scheduler based on validation loss
        scheduler.step(val_loss)

        # Save the best model and stop if no improvement
        if val_auprc > best_val_auprc:
            best_val_auprc = val_auprc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        val_logits = model(x_val_tensor)
        val_probs = torch.sigmoid(val_logits).cpu().numpy().ravel()

    return val_probs, float(best_val_auprc)

def run_mlp_cv(x_dev: pd.DataFrame, y_dev: np.ndarray, n_splits: int = 5, seed: int = 42) -> dict[str, object]:
    """Run 5-fold CV for MLP."""
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    fold_auprc: list[float] = []
    fold_f1: list[float] = []
    fold_precision: list[float] = []
    fold_recall: list[float] = []
    fold_threshold: list[float] = []

    oof_probs = np.zeros(len(y_dev), dtype=float)

    for fold, (train_idx, val_idx) in enumerate(cv.split(x_dev, y_dev), start=1):
        logger.info(f"[MLP] Fold {fold}/{n_splits}")

        x_train_fold, y_train_fold = x_dev.iloc[train_idx].copy(), y_dev[train_idx]
        x_val_fold, y_val_fold = x_dev.iloc[val_idx].copy(), y_dev[val_idx]

        imputer, scaler, x_train_fold_scaled = fit_preprocessor(x_train_fold, seed)
        x_val_fold_scaled = transform_fold(x_val_fold, imputer, scaler)

        val_probs, val_auprc = train_mlp_fold(
            x_train_fold_scaled,
            y_train_fold,
            x_val_fold_scaled,
            y_val_fold,
            epochs=500,
            lr=1e-3,
            seed=seed,
        )
        oof_probs[val_idx] = val_probs

        best_threshold, best_f1, best_precision, best_recall = find_best_threshold(y_val_fold, val_probs)

        fold_auprc.append(float(val_auprc))
        fold_f1.append(float(best_f1))
        fold_precision.append(float(best_precision))
        fold_recall.append(float(best_recall))
        fold_threshold.append(float(best_threshold))

        logger.info(
            f"[MLP] Fold {fold} | "
            f"AUPRC: {val_auprc:.4f} | "
            f"F1: {best_f1:.4f} | "
            f"Precision: {best_precision:.4f} | "
            f"Recall: {best_recall:.4f} | "
            f"Threshold: {best_threshold:.4f}",
        )

    return {
        "fold_auprc": fold_auprc,
        "fold_f1": fold_f1,
        "fold_precision": fold_precision,
        "fold_recall": fold_recall,
        "fold_threshold": fold_threshold,
        "mean_auprc": float(np.mean(fold_auprc)),
        "mean_f1": float(np.mean(fold_f1)),
        "mean_precision": float(np.mean(fold_precision)),
        "mean_recall": float(np.mean(fold_recall)),
        "mean_threshold": float(np.mean(fold_threshold)),
        "oof_probs": oof_probs,
    }

def save_cv_results(results: dict[str, dict[str, object]], y_dev: np.ndarray) -> None:
    """Save fold metrics and OOF predictions."""
    summary: dict[str, dict[str, object]] = {}

    for model_name, model_results in results.items():
        np.save(MODELS_DIR / f"{model_name}_oof_probs.npy", model_results["oof_probs"])

        summary[model_name] = {
            "fold_auprc": model_results["fold_auprc"],
            "fold_f1": model_results["fold_f1"],
            "fold_precision": model_results["fold_precision"],
            "fold_recall": model_results["fold_recall"],
            "fold_threshold": model_results["fold_threshold"],
            "mean_auprc": model_results["mean_auprc"],
            "mean_f1": model_results["mean_f1"],
            "mean_precision": model_results["mean_precision"],
            "mean_recall": model_results["mean_recall"],
            "mean_threshold": model_results["mean_threshold"],
        }

    np.save(MODELS_DIR / "y_dev.npy", y_dev)

    with (METRICS_DIR / "cv_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"CV results saved to {METRICS_DIR}")

def plot_cv_boxplots(results: dict[str, dict[str, object]]) -> None:
    """Plot boxplots of fold AUPRC for each model."""
    df = pd.DataFrame({
        "Decision Tree": results["dt"]["fold_auprc"],
        "Random Forest": results["rf"]["fold_auprc"],
        "MLP": results["mlp"]["fold_auprc"],
    })

    plt.figure(figsize=(8, 6))
    df.boxplot()
    plt.ylabel("AUPRC")
    plt.title("5-Fold CV Performance")
    plt.grid(visible=True, linestyle="--", alpha=0.5)

    output_dir = METRICS_DIR / "cv_boxplots_auprc.png"
    plt.savefig(output_dir, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"CV boxplots saved to {output_dir}")

if __name__ == "__main__":
    """Main function to run 5-fold CV for all models."""
    set_seed(42)

    x_dev, y_dev = load_data()

    dt_results = run_dt_cv(x_dev, y_dev, n_splits=5, seed=42)
    rf_results = run_rf_cv(x_dev, y_dev, n_splits=5, seed=42)
    mlp_results = run_mlp_cv(x_dev, y_dev, n_splits=5, seed=42)

    results = {
        "dt": dt_results,
        "rf": rf_results,
        "mlp": mlp_results,
    }

    save_cv_results(results, y_dev)
    plot_cv_boxplots(results)

    logger.info("5-fold CV completed for all models.")
