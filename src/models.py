"""Model training module for Random Forest and MLP."""
import logging
import pickle
from pathlib import Path

import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV
from torch import nn, optim

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path("../data/processed")
MODELS_DIR = Path("../models")
MODELS_DIR.mkdir(exist_ok=True)

def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Load preprocessed datasets."""
    x_train = pd.read_csv(DATA_DIR / "x_train.csv")
    x_test = pd.read_csv(DATA_DIR / "x_test.csv")
    y_train = pd.read_csv(DATA_DIR / "y_train.csv").to_numpy().ravel()
    y_test = pd.read_csv(DATA_DIR / "y_test.csv").to_numpy().ravel()

    return x_train, x_test, y_train, y_test

# --- Random Forest Training ---

def train_rf(x_train: pd.DataFrame, x_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series) -> None:
    """Train a Random Forest Classifier."""
    logger.info("Starting Random Forest training...")

    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5],
        "class_weight": ["balanced", None],
    }

    rf_model = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(rf_model, param_grid, cv=5, scoring="f1")
    grid_search.fit(x_train, y_train)

    best_rf = grid_search.best_estimator_
    preds = best_rf.predict(x_test)

    logger.info(f"Best Random Forest Params: {grid_search.best_params_}") # noqa: G004
    logger.info(f"Random Forest Accuracy: {accuracy_score(y_test, preds):.4f}")  # noqa: G004
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
            nn.Sigmoid(),
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

    model = MLP(x_train.shape[1])
    criterion = nn.BCELoss()
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
            logger.info(f"Epoch [{epoch + 1}/200], Loss: {loss.item():.4f}")  # noqa: G004

    model.eval()
    with torch.no_grad():
        test_outputs = model(x_test_tensor)
        preds = (test_outputs > 0.5).float().numpy()
        logger.info(f"MLP Test Accuracy: {accuracy_score(y_test, preds):.4f}")  # noqa: G004

    torch.save(model.state_dict(), MODELS_DIR / "mlp_model.pth")
    logger.info(f"MLP saved to {MODELS_DIR / 'mlp_model.pth'}")  # noqa: G004

    return model

if __name__ == "__main__":
    try:
        x_train, x_test, y_train, y_test = load_data()

        train_rf(x_train, x_test, y_train, y_test)
        logger.info("-" * 30)
        train_mlp(x_train, y_train, x_test, y_test)
    except Exception as e:
        logger.exception(f"An error occurred during training: {e}")  # noqa: G004

