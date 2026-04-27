"""Model training module for Random Forest and MLP."""
import json
import logging
import pickle
import random
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, classification_report, f1_score
from sklearn.tree import DecisionTreeClassifier
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path("../data/processed")
MODELS_DIR = Path("../models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)
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
            params: None,
        ) -> RandomForestClassifier:
    """Train a Random Forest Classifier using training data."""
    logger.info("Starting Random Forest training...")

    if params is None:
        params = {}

    rf_model = RandomForestClassifier(
        **params,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    )

    rf_model.fit(x_train, y_train)

    with (MODELS_DIR / "rf_model.pkl").open("wb") as f:
        pickle.dump(rf_model, f)
    logger.info(f"RF model saved to {MODELS_DIR / 'rf_model.pkl'}")

    return rf_model

# --- Decision Tree (DT) ---

def train_dt(x_train: pd.DataFrame, y_train: np.ndarray, params: None) -> DecisionTreeClassifier:
    """Train a Decision Tree Classifier using training data."""
    logger.info("Starting Decision Tree training...")

    if params is None:
        params = {}

    dt_model = DecisionTreeClassifier(
        **params,
        class_weight="balanced",
        random_state=42,
    )

    dt_model.fit(x_train, y_train)

    with (MODELS_DIR / "dt_model.pkl").open("wb") as f:
        pickle.dump(dt_model, f)
    logger.info(f"DT model saved to {MODELS_DIR / 'dt_model.pkl'}")

    return dt_model

# --- Multi-Layer Perceptron (MLP) ---

class MLP(nn.Module):
    """Multi-Layer Perceptron for binary classification."""

    def __init__(
            self,
            input_size: int,
            hidden_layers: list[int] | None = None,
            dropout: float = 0.1,
        ) -> None:
        """Initialise the MLP model."""
        super().__init__()

        if hidden_layers is None:
            hidden_layers = [128, 64, 32]

        layers = []
        prev_size = input_size

        for hidden_size in hidden_layers:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_size = hidden_size
        layers.append(nn.Linear(prev_size, 1))


        self.layers = nn.Sequential(*layers)

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
        weight_decay: float = 1e-3,
        dropout: float = 0.1,
        batch_size: int = 64,
        hidden_layers: list[int] | None = None,
        save_path: Path = MODELS_DIR / "mlp_model.pth",
        trial: optuna.Trial | None = None,
    ) -> MLP:
    """Train a Multi-Layer Perceptron model using training data."""
    logger.info("Starting MLP training...")

    model = MLP(
        input_size=x_train.shape[1],
        hidden_layers=hidden_layers,
        dropout=dropout,
    ).to(device)

    # 1. Weighted Loss: handle class imbalance
    pos = float((y_train == 1).sum())
    neg = float((y_train == 0).sum())
    pos_weight = torch.tensor([neg / pos], dtype=torch.float32, device=device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimiser = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # 2. LR Scheduler: reduces learning rate on plateau of validation loss
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimiser, mode="min", factor=0.5, patience=6, min_lr=1e-5)

    # Convert data to tensors
    x_train_tensor = torch.tensor(x_train.values, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)

    x_val_tensor = torch.tensor(x_val.values, dtype=torch.float32, device=device)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32, device=device).view(-1, 1)

    train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    best_val_auprc = -float("inf")
    patience_counter = 0
    early_stop_patience = 20

    torch.save(model.state_dict(), save_path)

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

            # Safe AUPRC
            try:
                val_auprc = average_precision_score(y_val, val_probs)
            except Exception:
                val_auprc = 0.0

        if trial is not None:
            trial.report(val_auprc, step=epoch)

            if trial.should_prune():
                save_path.unlink(missing_ok=True)
                raise optuna.TrialPruned


        # Step the scheduler based on validation loss
        scheduler.step(val_loss)

        # Save the best model and stop if no improvement
        if val_auprc > best_val_auprc:
            best_val_auprc = val_auprc
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
        else:
            patience_counter += 1

            if epoch % 10 == 0:
                logger.info(
                    f"Epoch {epoch} | "
                    f"Train Loss: {train_loss:.4f} | "
                    f"Val Loss: {val_loss.item():.4f} | "
                    f"Val AUPRC: {val_auprc:.4f}",
                )

            if patience_counter >= early_stop_patience:
                logger.info(f"Early stopping at epoch {epoch}. ")
                break

    # Load the best model before returning
    model.load_state_dict(torch.load(save_path, weights_only=True))

    return model

# Evaluation helpers

def predict_probs(model: nn.Module, x: pd.DataFrame, is_pytorch: bool = False) -> np.ndarray:
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

def load_hyperparameters() -> dict:
    """Load hyperparameters from JSON file."""
    with (METRICS_DIR / "hyperparameters.json").open() as f:
        return json.load(f)
