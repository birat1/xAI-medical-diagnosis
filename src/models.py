"""Model training module for Random Forest and MLP."""
import json
import logging
import pickle
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import randint
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, classification_report, f1_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path("../data/processed")
MODELS_DIR = Path("../models")
MODELS_DIR.mkdir(exist_ok=True)
METRICS_DIR = Path("../results/metrics")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """Load preprocessed train/val/test datasets."""
    x_train = pd.read_csv(DATA_DIR / "x_train.csv")
    y_train = pd.read_csv(DATA_DIR / "y_train.csv").to_numpy().ravel()

    x_val = pd.read_csv(DATA_DIR / "x_val.csv")
    y_val = pd.read_csv(DATA_DIR / "y_val.csv").to_numpy().ravel()

    x_test = pd.read_csv(DATA_DIR / "x_test.csv")
    y_test = pd.read_csv(DATA_DIR / "y_test.csv").to_numpy().ravel()

    return x_train, x_val, x_test, y_train, y_val, y_test

def set_seed(seed: int = 42) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.default_rng(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info(f"Global seed set to {seed}")

# --- Random Forest (RF) ---

def train_rf(
            x_train: pd.DataFrame,
            y_train: np.ndarray,
        ) -> RandomForestClassifier:
    """Train a Random Forest Classifier using training data."""
    logger.info("Starting Random Forest training with 5-Fold CV...")

    """
    best_params = {
        "n_estimators": randint(100, 500),
        "max_depth": randint(5, 20),
        "min_samples_split": randint(2, 20),
        "min_samples_leaf": randint(1, 10),
    }
    """

    rf_model = RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        min_samples_leaf=5,
        min_samples_split=8,
        class_weight="balanced_subsample",
        random_state=42,
    )

    """
    random_search = RandomizedSearchCV(
        estimator=rf_model,
        param_distributions=best_params,
        n_iter=100,
        cv=5,
        scoring="average_precision",
        n_jobs=-1,
        verbose=2,
        random_state=42,
    )
    """

    rf_model.fit(x_train, y_train)


    logger.info("-" * 50)
    # logger.info(f"Best RF Hyperparameters: {random_search.best_params_}")
    logger.info(f"Best Random Forest CV Score: {rf_model.score(x_train, y_train):.4f}")

    with (MODELS_DIR / "rf_model.pkl").open("wb") as f:
        pickle.dump(rf_model, f)
    logger.info("RF model saved to ../models/rf_model.pkl")

    return rf_model


# --- Decision Tree (DT) ---

def train_dt(x_train: pd.DataFrame, y_train: np.ndarray) -> DecisionTreeClassifier:
    """Train a Decision Tree Classifier using training data."""
    logger.info("Starting Decision Tree training with 5-Fold CV...")

    """
    best_params = {
        "max_depth": [3, 5, 7, 10, None],
        "min_samples_split": randint(10, 50),
        "min_samples_leaf": randint(5, 20),
        "criterion": ["gini", "entropy"],
        "ccp_alpha": [0.005, 0.01],
    }
    """

    dt_model = DecisionTreeClassifier(
        max_depth=10,
        min_samples_split=8,
        min_samples_leaf=5,
        criterion="entropy",
        ccp_alpha=0.001,
        class_weight="balanced",
        random_state=42,
    )

    """
    random_search = RandomizedSearchCV(
        estimator=dt_model,
        param_distributions=best_params,
        n_iter=100,
        cv=5,
        scoring="average_precision",
        n_jobs=-1,
        verbose=2,
        random_state=42,
    )
    """

    dt_model.fit(x_train, y_train)

    logger.info("-" * 50)
    # logger.info(f"Best DT Hyperparameters: {random_search.best_params_}")
    logger.info(f"Best Decision Tree CV Score: {dt_model.score(x_train, y_train):.4f}")

    # Save the best model
    with (MODELS_DIR / "dt_model.pkl").open("wb") as f:
        pickle.dump(dt_model, f)
    logger.info("DT model saved to ../models/dt_model.pkl")

    return dt_model


# --- Multi-Layer Perceptron (MLP) ---

class MLP(nn.Module):
    """Multi-Layer Perceptron for binary classification."""

    def __init__(self, input_size: int) -> None:
        """Initialise the MLP model."""
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),

            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Define the forward pass."""
        return self.layers(x)

def train_mlp(  # noqa: PLR0913
        x_train: pd.DataFrame,
        y_train: np.ndarray,
        x_val: pd.DataFrame,
        y_val: np.ndarray,
        epochs: int = 200,
        lr: float = 1e-3,
    ) -> MLP:
    """Train a Multi-Layer Perceptron model using training data."""
    logger.info("Starting MLP training...")

    model = MLP(x_train.shape[1]).to(device)

    # 1. Weighted Loss: handle class imbalance
    pos = float((y_train == 1).sum())
    neg = float((y_train == 0).sum())
    pos_weight = torch.tensor([neg / pos], dtype=torch.float32, device=device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimiser = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-3)

    # 2. LR Scheduler: reduces learning rate on plateau of validation loss
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimiser, mode="min", factor=0.5, patience=6, min_lr=1e-3)

    # Convert data to tensors
    x_train_tensor = torch.tensor(x_train.values, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)

    x_val_tensor = torch.tensor(x_val.values, dtype=torch.float32, device=device)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32, device=device).view(-1, 1)

    train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, drop_last=False)

    best_val_auprc = -float("inf")
    best_val_loss = float("inf")
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
            best_val_loss = float(val_loss.item())
            patience_counter = 0
            torch.save(model.state_dict(), MODELS_DIR / "mlp_model.pth")
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                logger.info(
                    f"Early stopping at epoch {epoch}. "
                    f"Best Val AUPRC: {best_val_auprc:.4f} | "
                    f"Val Loss: {best_val_loss:.4f}",
                )
                break

        if epoch % 20 == 0:
            logger.info(
                f"Epoch [{epoch}/{epochs}] "
                f"Train Loss: {loss.item():.4f} | "
                f"Val Loss: {val_loss.item():.4f} | "
                f"Val AUPRC: {val_auprc:.4f}",
            )

    logger.info("MLP training completed and best model saved to ../models/mlp_model.pth")
    # Load the best model before returning
    model.load_state_dict(torch.load(MODELS_DIR / "mlp_model.pth", weights_only=True))
    return model


# Evaluation helpers

def predict_probs(model: nn.Module, x: pd.DataFrame, is_pytorch: bool) -> np.ndarray:
    """Predict probabilities using the given model."""
    if is_pytorch:
        model.eval()
        with torch.no_grad():
            logits = model(torch.tensor(x.values, dtype=torch.float32).to(device))
            return torch.sigmoid(logits).cpu().numpy().ravel()
    return model.predict_proba(x)[:, 1]

def evaluate(y_true: np.ndarray, probs: np.ndarray, threshold: float) -> tuple[float, str]:
    """Evaluate model performance at a given threshold."""
    preds = (probs >= threshold).astype(int)
    return float(f1_score(y_true, preds)), classification_report(y_true, preds)

def load_thresholds() -> dict[str, float] | None:
    """Load optimal thresholds from JSON file."""
    if not (METRICS_DIR / "thresholds.json").exists():
        return None
    with (METRICS_DIR / "thresholds.json").open("r") as f:
        data = json.load(f)
        # expected keys: rf_threshold, dt_threshold, mlp_threshold
    if "rf_threshold" in data and "dt_threshold" in data and "mlp_threshold" in data:
        return {
            "rf": float(data["rf_threshold"]),
            "dt": float(data["dt_threshold"]),
            "mlp": float(data["mlp_threshold"]),
        }
    return None


if __name__ == "__main__":
    set_seed(42)
    logger.info("-" * 50)

    try:
        x_train, x_val, x_test, y_train, y_val, y_test = load_data()

        # Train models using training data only
        rf_model = train_rf(x_train, y_train)
        logger.info("-" * 50)
        dt_model = train_dt(x_train, y_train)
        logger.info("-" * 50)
        mlp_model = train_mlp(x_train, y_train, x_val, y_val , epochs=500, lr=1e-3)
        logger.info("-" * 50)

        rf_val_probs = predict_probs(rf_model, x_val, is_pytorch=False)
        mlp_val_probs = predict_probs(mlp_model, x_val, is_pytorch=True)
        dt_val_probs = predict_probs(dt_model, x_val, is_pytorch=False)
        rf_test_probs = predict_probs(rf_model, x_test, is_pytorch=False)
        mlp_test_probs = predict_probs(mlp_model, x_test, is_pytorch=True)
        dt_test_probs = predict_probs(dt_model, x_test, is_pytorch=False)

        rf_val_f1, rf_val_rep = evaluate(y_val, rf_val_probs, threshold=0.5)
        dt_val_f1, dt_val_rep = evaluate(y_val, dt_val_probs, threshold=0.5)
        mlp_val_f1, mlp_val_rep = evaluate(y_val, mlp_val_probs, threshold=0.5)

        logger.info("Validation Set (threshold=0.5)")
        logger.info(f"Random Forest F1: {rf_val_f1:.4f}")
        logger.info(rf_val_rep)
        logger.info(f"Decision Tree F1: {dt_val_f1:.4f}")
        logger.info(dt_val_rep)
        logger.info(f"Multi-Layer Perceptron F1: {mlp_val_f1:.4f}")
        logger.info(mlp_val_rep)
        logger.info("-" * 50)

        rf_test_f1, rf_test_rep = evaluate(y_test, rf_test_probs, threshold=0.5)
        dt_test_f1, dt_test_rep = evaluate(y_test, dt_test_probs, threshold=0.5)
        mlp_test_f1, mlp_test_rep = evaluate(y_test, mlp_test_probs, threshold=0.5)

        logger.info("Test Set (threshold=0.5)")
        logger.info(f"Random Forest F1: {rf_test_f1:.4f}")
        logger.info(rf_test_rep)
        logger.info(f"Decision Tree F1: {dt_test_f1:.4f}")
        logger.info(dt_test_rep)
        logger.info(f"Multi-Layer Perceptron F1: {mlp_test_f1:.4f}")
        logger.info(mlp_test_rep)
        logger.info("-" * 50)

        thresholds = load_thresholds()
        if thresholds is not None:
            logger.info("Test Set (tuned thresholds from thresholds.json)")

            rf_tuned_f1, rf_tuned_rep = evaluate(y_test, rf_test_probs, threshold=thresholds["rf"])
            dt_tuned_f1, dt_tuned_rep = evaluate(y_test, dt_test_probs, threshold=thresholds["dt"])
            mlp_tuned_f1, mlp_tuned_rep = evaluate(y_test, mlp_test_probs, threshold=thresholds["mlp"])

            logger.info(f"Random Forest F1 (threshold={thresholds['rf']:.4f}): {rf_tuned_f1:.4f}")
            logger.info(rf_tuned_rep)
            logger.info(f"Decision Tree F1 (threshold={thresholds['dt']:.4f}): {dt_tuned_f1:.4f}")
            logger.info(dt_tuned_rep)
            logger.info(f"Multi-Layer Perceptron F1 (threshold={thresholds['mlp']:.4f}): {mlp_tuned_f1:.4f}")
            logger.info(mlp_tuned_rep)
        else:
            logger.info("No tuned thresholds found. Please run optimise_threshold.py first.")
    except Exception as e:
        logger.exception(f"An error occurred during training: {e}")
