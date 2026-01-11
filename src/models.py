"""Model training module for Random Forest and MLP."""
import logging
import pickle
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import GridSearchCV
from torch import nn, optim

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path("../data/processed")
MODELS_DIR = Path("../models")
MODELS_DIR.mkdir(exist_ok=True)

RF_THRESHOLD = 0.3775
MLP_THRESHOLD = 0.5781

def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Load preprocessed datasets."""
    x_train = pd.read_csv(DATA_DIR / "x_train.csv")
    x_test = pd.read_csv(DATA_DIR / "x_test.csv")
    y_train = pd.read_csv(DATA_DIR / "y_train.csv").to_numpy().ravel()
    y_test = pd.read_csv(DATA_DIR / "y_test.csv").to_numpy().ravel()

    return x_train, x_test, y_train, y_test

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

# --- Random Forest Training ---

def train_rf(x_train: pd.DataFrame, x_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series) -> None:
    """Train a Random Forest Classifier."""
    logger.info("Starting Random Forest training...")

    param_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [None, 10, 15],
        "min_samples_leaf": [1, 2, 4],
        "class_weight": ["balanced", "balanced_subsample"],
    }

    rf_model = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(rf_model, param_grid, cv=5, scoring="average_precision")
    grid_search.fit(x_train, y_train)

    best_rf = grid_search.best_estimator_
    preds = best_rf.predict(x_test)

    logger.info(f"Best Random Forest Params: {grid_search.best_params_}")
    logger.info(f"Random Forest Accuracy: {accuracy_score(y_test, preds):.4f}")
    logger.info("Classification Report:")
    logger.info(classification_report(y_test, preds))

    with (MODELS_DIR / "rf_model.pkl").open("wb") as f:
        pickle.dump(best_rf, f)
    logger.info("Random Forest model saved to ../models/rf_model.pkl")

    return best_rf

# --- Multi-Layer Perceptron Training ---

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

def train_mlp(x_train: pd.DataFrame, y_train: pd.Series, x_test: pd.DataFrame, y_test: pd.Series) -> MLP:
    """Train a Multi-Layer Perceptron model."""
    logger.info("Starting MLP training...")

    x_train_tensor = torch.tensor(x_train.values, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    x_test_tensor = torch.tensor(x_test.values, dtype=torch.float32)

    num_neg = (y_train == 0).sum()
    num_pos = (y_train == 1).sum()
    pos_weight = torch.tensor([num_neg / num_pos], dtype=torch.float32)

    model = MLP(x_train.shape[1])
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimiser = optim.Adam(model.parameters(), lr=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimiser, "min", patience=10)

    model.train()
    for epoch in range(200):
        optimiser.zero_grad()
        outputs = model(x_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimiser.step()
        scheduler.step(loss.detach())

        if (epoch + 1) % 50 == 0:
            logger.info(f"Epoch [{epoch + 1}/200], Loss: {loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        test_outputs = model(x_test_tensor)
        preds = (test_outputs > 0.5).float().numpy()  # noqa: PLR2004
        logger.info(f"MLP Test Accuracy: {accuracy_score(y_test, preds):.4f}")

    torch.save(model.state_dict(), MODELS_DIR / "mlp_model.pth")
    logger.info(f"MLP saved to {MODELS_DIR / 'mlp_model.pth'}")

    return model

def evaluate_model_performance(
        model: nn.Module,
        x_test: pd.DataFrame,
        y_test: pd.Series,
        threshold: float,
        is_pytorch: bool = False,  # noqa: FBT001, FBT002
    ) -> float:
    """Evaluate model performance using a specified threshold."""
    if is_pytorch:
        model.eval()
        with torch.no_grad():
            logits = model(torch.tensor(x_test.values, dtype=torch.float32))
            probs = torch.sigmoid(logits).numpy().ravel()
    else:
        probs = model.predict_proba(x_test)[:, 1]

    preds = (probs >= threshold).astype(int)

    report = classification_report(y_test, preds)
    f1 = f1_score(y_test, preds)
    return f1, report

if __name__ == "__main__":
    set_seed(42)
    try:
        x_train, x_test, y_train, y_test = load_data()

        rf_model = train_rf(x_train, x_test, y_train, y_test)
        logger.info("-" * 50)
        mlp_model = train_mlp(x_train, y_train, x_test, y_test)
        logger.info("-" * 50)

        rf_f1, rf_report = evaluate_model_performance(
                                                    rf_model,
                                                    x_test,
                                                    y_test,
                                                    threshold=RF_THRESHOLD,
                                                    is_pytorch=False,
                                                    )
        mlp_f1, mlp_report = evaluate_model_performance(
                                                    mlp_model,
                                                    x_test,
                                                    y_test,
                                                    threshold=MLP_THRESHOLD,
                                                    is_pytorch=True,
                                                    )

        logger.info(f"Random Forest F1 Score: {rf_f1:.4f}")
        logger.info(rf_report)
        logger.info(f"MLP F1 Score: {mlp_f1:.4f}")
        logger.info(mlp_report)
    except Exception as e:
        logger.exception(f"An error occurred during training: {e}")

