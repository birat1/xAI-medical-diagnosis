"""Final training on the entire development set."""
import json
import logging
import pickle
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

from models import MLP

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path("../data/processed")
MODELS_DIR = Path("../models")
METRICS_DIR = Path("../results/metrics")
METRICS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR = Path("../results")

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

def load_data() -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray]:
    """Load development data for 5-fold CV."""
    x_dev = pd.read_csv(DATA_DIR / "x_dev.csv")
    y_dev = pd.read_csv(DATA_DIR / "y_dev.csv").to_numpy().ravel()

    x_test = pd.read_csv(DATA_DIR / "x_test.csv")
    y_test = pd.read_csv(DATA_DIR / "y_test.csv").to_numpy().ravel()

    return x_dev, y_dev, x_test, y_test

def fit_preprocessor(x_dev: pd.DataFrame, seed: int = 42) -> tuple[IterativeImputer, StandardScaler, np.ndarray]:
    """Fit imputer and scaler on training fold."""
    imputer = IterativeImputer(random_state=seed)
    scaler = StandardScaler()

    x_dev_imputed = imputer.fit_transform(x_dev)
    x_dev_scaled = scaler.fit_transform(x_dev_imputed)

    return imputer, scaler, x_dev_scaled

def transform_fold(x_data: pd.DataFrame, imputer: IterativeImputer, scaler: StandardScaler) -> np.ndarray:
    """Transform fold using fitted preprocessor."""
    x_imputed = imputer.transform(x_data)
    x_scaled = scaler.transform(x_imputed)

    return x_scaled

def save_preprocessor(imputer: IterativeImputer, scaler: StandardScaler, columns: list[str], seed: int = 42) -> None:
    """Save fitted preprocessor artifacts."""
    artifacts = {
        "imputer": imputer,
        "scaler": scaler,
        "columns": columns,
        "seed": seed,
    }

    with (MODELS_DIR / "preprocessor.pkl").open("wb") as f:
        pickle.dump(artifacts, f)

    logger.info(f"Preprocessor artifacts saved to {MODELS_DIR / 'preprocessor.pkl'}")

###
###
###

def train_final_dt(x_dev_scaled: np.ndarray, y_dev: np.ndarray, seed: int = 42) -> DecisionTreeClassifier:
    """Train final Decision Tree on entire development set."""
    logger.info("Training final Decision Tree on entire development set...")

    model = DecisionTreeClassifier(
        max_depth=10,
        min_samples_split=2,
        min_samples_leaf=1,
        criterion="entropy",
        ccp_alpha=0.0005,
        class_weight="balanced",
        random_state=42,
    )
    model.fit(x_dev_scaled, y_dev)

    with (MODELS_DIR / "dt_model.pkl").open("wb") as f:
        pickle.dump(model, f)
    logger.info(f"Final DT model saved to {MODELS_DIR / 'dt_model.pkl'}")

    return model

def train_final_rf(x_dev_scaled: np.ndarray, y_dev: np.ndarray, seed: int = 42) -> RandomForestClassifier:
    """Train final Random Forest on entire development set."""
    logger.info("Training final Random Forest on entire development set...")

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_leaf=1,
        min_samples_split=2,
        class_weight="balanced",
        max_features="log2",
        random_state=42,
    )
    model.fit(x_dev_scaled, y_dev)

    with (MODELS_DIR / "rf_model.pkl").open("wb") as f:
        pickle.dump(model, f)
    logger.info(f"Final RF model saved to {MODELS_DIR / 'rf_model.pkl'}")

    return model

def train_final_mlp(  # noqa: PLR0913, PLR0915
        x_dev_scaled: np.ndarray,
        y_dev: np.ndarray,
        epochs: int = 500,
        lr: float = 1e-3,
        seed: int = 42,
    ) -> MLP:
    """Train final MLP on entire development set."""
    logger.info("Training final MLP on entire development set...")

    set_seed(seed)

    x_train, x_val, y_train, y_val = train_test_split(
        x_dev_scaled, y_dev,
        test_size=0.1,
        random_state=seed,
        stratify=y_dev,
    )

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
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                logger.info(
                    f"Early stopping at epoch {epoch}. "
                    f"Best Val AUPRC: {best_val_auprc:.4f} | "
                    f"Val Loss: {best_val_loss:.4f}",
                )
                break

        if epoch % 10 == 0:
            logger.info(
                f"Epoch [{epoch}/{epochs}] | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val AUPRC: {val_auprc:.4f}",
            )

    if best_state is not None:
        model.load_state_dict(best_state)

    torch.save(model.state_dict(), MODELS_DIR / "mlp_model.pth")
    logger.info(f"Final MLP model saved to {MODELS_DIR / 'mlp_model.pth'}")

    return model

def predict_probs(model: object, x_scaled: np.ndarray, is_pytorch: bool) -> np.ndarray:
    """Predict probabilities for final evaluation checks."""
    if is_pytorch:
        model.eval()
        with torch.no_grad():
            logits = model(torch.tensor(x_scaled, dtype=torch.float32).to(device))
            return torch.sigmoid(logits).cpu().numpy().ravel()
    return model.predict_proba(x_scaled)[:, 1]

def save_training_summary(summary: dict[str, object]) -> None:
    """Save final training summary metrics."""
    with (METRICS_DIR / "final_training_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Final training summary saved to {METRICS_DIR / 'final_training_summary.json'}")

if __name__ == "__main__":
    """Main function to orchestrate final training and evaluation."""
    set_seed(42)

    x_dev, y_dev, x_test, y_test = load_data()

    logger.info(f"Loaded data | Dev size: {len(x_dev)} | Test size: {len(x_test)}")

    imputer, scaler, x_dev_scaled = fit_preprocessor(x_dev, seed=42)
    x_test_scaled = transform_fold(x_test, imputer, scaler)

    save_preprocessor(imputer, scaler, columns=list(x_dev.columns), seed=42)

    dt_model = train_final_dt(x_dev_scaled, y_dev, seed=42)
    rf_model = train_final_rf(x_dev_scaled, y_dev, seed=42)
    mlp_model = train_final_mlp(x_dev_scaled, y_dev, epochs=500, lr=1e-3, seed=42)

    dt_test_probs = predict_probs(dt_model, x_test_scaled, is_pytorch=False)
    rf_test_probs = predict_probs(rf_model, x_test_scaled, is_pytorch=False)
    mlp_test_probs = predict_probs(mlp_model, x_test_scaled, is_pytorch=True)

    summary = {
        "dev_size": len(x_dev),
        "test_size": len(x_test),
        "feature_count": x_dev.shape[1],
        "dt_test_auprc": float(average_precision_score(y_test, dt_test_probs)),
        "rf_test_auprc": float(average_precision_score(y_test, rf_test_probs)),
        "mlp_test_auprc": float(average_precision_score(y_test, mlp_test_probs)),
        "saved_models": {
            "dt": str(MODELS_DIR / "dt_model.pkl"),
            "rf": str(MODELS_DIR / "rf_model.pkl"),
            "mlp": str(MODELS_DIR / "mlp_model.pth"),
            "preprocessor": str(MODELS_DIR / "preprocessor.pkl"),
        },
    }

    save_training_summary(summary)
    logger.info("Final training completed successfully.")
