"""Model training module for Random Forest and MLP."""
import json
import logging
import pickle
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import GridSearchCV
from torch import nn, optim

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path("../data/processed")
MODELS_DIR = Path("../models")
MODELS_DIR.mkdir(exist_ok=True)
METRICS_DIR = Path("../results/metrics")

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

# --- Random Forest ---

def train_rf(
            x_train: pd.DataFrame,
            y_train: np.ndarray,
        ) -> RandomForestClassifier:
    """Train a Random Forest Classifier using training data."""
    logger.info("Starting Random Forest training with 5-Fold CV...")

    param_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [None, 10, 15],
        "min_samples_leaf": [1, 2, 4],
        "class_weight": ["balanced", "balanced_subsample"],
    }

    rf_model = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(rf_model, param_grid, cv=5, scoring="average_precision", n_jobs=-1)
    grid_search.fit(x_train, y_train)

    best_rf = grid_search.best_estimator_
    logger.info(f"Best Random Forest Parameters: {grid_search.best_params_}")
    logger.info(f"Best Random Forest CV Score: {grid_search.best_score_:.4f}")

    with (MODELS_DIR / "rf_model.pkl").open("wb") as f:
        pickle.dump(best_rf, f)
    logger.info("RF model saved to ../models/rf_model.pkl")

    return best_rf

# --- Multi-Layer Perceptron (MLP) ---

class MLP(nn.Module):
    """Multi-Layer Perceptron for binary classification."""

    def __init__(self, input_size: int) -> None:
        """Initialise the MLP model."""
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Linear(16, 1),
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
        lr: float = 1e-2,
    ) -> MLP:
    """Train a Multi-Layer Perceptron model using training data."""
    logger.info("Starting MLP training...")

    # 1. Weighted Loss: handle class imbalance
    model = MLP(x_train.shape[1])
    pos_weight = torch.tensor([(y_train == 0).sum() / (y_train == 1).sum()], dtype=torch.float32)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimiser = optim.Adam(model.parameters(), lr=lr)

    # 2. LR Scheduler: reduces learning rate on plateau of validation loss
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimiser, mode="min", factor=0.1, patience=10)

    # Convert data to tensors
    x_train_tensor = torch.tensor(x_train.values, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    x_val_tensor = torch.tensor(x_val.values, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)

    best_val_loss = float("inf")
    early_stop_patience = 20
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        # Training phase
        model.train()
        optimiser.zero_grad()
        logits = model(x_train_tensor)
        loss = criterion(logits, y_train_tensor)
        loss.backward()
        optimiser.step()

        # 3. Validation phase for early stopping and LR scheduling
        model.eval()
        with torch.no_grad():
            val_logits = model(x_val_tensor)
            val_loss = nn.BCEWithLogitsLoss()(val_logits, y_val_tensor)

        # Step the scheduler based on validation loss
        scheduler.step(val_loss)

        # Save the best model and stop if no improvement
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), MODELS_DIR / "mlp_model.pth")
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                logger.info(f"Early stopping at epoch {epoch}. Best Val Loss: {best_val_loss.item():.4f}")
                break

        if epoch % 20 == 0:
            logger.info(f"Epoch [{epoch}/{epochs}], Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}")

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
            logits = model(torch.tensor(x.values, dtype=torch.float32))
            return torch.sigmoid(logits).numpy().ravel()
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
        # expected keys: rf_threshold, mlp_threshold
    if "rf_threshold" in data and "mlp_threshold" in data:
        return {"rf": float(data["rf_threshold"]), "mlp": float(data["mlp_threshold"])}
    return None

if __name__ == "__main__":
    set_seed(42)
    try:
        x_train, x_val, x_test, y_train, y_val, y_test = load_data()

        # Train models using training data only
        rf_model = train_rf(x_train, y_train)
        logger.info("-" * 50)
        mlp_model = train_mlp(x_train, y_train, x_val, y_val , epochs=500, lr=1e-2)
        logger.info("-" * 50)

        rf_val_probs = predict_probs(rf_model, x_val, is_pytorch=False)
        mlp_val_probs = predict_probs(mlp_model, x_val, is_pytorch=True)
        rf_test_probs = predict_probs(rf_model, x_test, is_pytorch=False)
        mlp_test_probs = predict_probs(mlp_model, x_test, is_pytorch=True)

        rf_val_f1, rf_val_rep = evaluate(y_val, rf_val_probs, threshold=0.5)
        mlp_val_f1, mlp_val_rep = evaluate(y_val, mlp_val_probs, threshold=0.5)

        logger.info("Validation Set (threshold=0.5)")
        logger.info(f"Random Forest F1: {rf_val_f1:.4f}")
        logger.info(rf_val_rep)
        logger.info(f"Multi-Layer Perceptron F1: {mlp_val_f1:.4f}")
        logger.info(mlp_val_rep)
        logger.info("-" * 50)

        rf_test_f1, rf_test_rep = evaluate(y_test, rf_test_probs, threshold=0.5)
        mlp_test_f1, mlp_test_rep = evaluate(y_test, mlp_test_probs, threshold=0.5)

        logger.info("Test Set (threshold=0.5)")
        logger.info(f"Random Forest F1: {rf_test_f1:.4f}")
        logger.info(rf_test_rep)
        logger.info(f"Multi-Layer Perceptron F1: {mlp_test_f1:.4f}")
        logger.info(mlp_test_rep)
        logger.info("-" * 50)

        thresholds = load_thresholds()
        if thresholds is not None:
            logger.info("Test Set (tuned thresholds from thresholds.json)")

            rf_tuned_f1, rf_tuned_rep = evaluate(y_test, rf_test_probs, threshold=thresholds["rf"])
            mlp_tuned_f1, mlp_tuned_rep = evaluate(y_test, mlp_test_probs, threshold=thresholds["mlp"])

            logger.info(f"Random Forest F1 (threshold={thresholds['rf']:.4f}): {rf_tuned_f1:.4f}")
            logger.info(rf_tuned_rep)
            logger.info(f"Multi-Layer Perceptron F1 (threshold={thresholds['mlp']:.4f}): {mlp_tuned_f1:.4f}")
            logger.info(mlp_tuned_rep)
        else:
            logger.info("-" * 50)
            logger.info("No tuned thresholds found. Please run optimise_threshold.py first.")
    except Exception as e:
        logger.exception(f"An error occurred during training: {e}")
