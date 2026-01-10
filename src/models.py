"""Model training module for Random Forest and MLP models."""
import logging
import pickle
from pathlib import Path

import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from torch import nn, optim

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
)

x_train = pd.read_csv(Path("../data/processed/x_train.csv"))
x_test = pd.read_csv(Path("../data/processed/x_test.csv"))
y_train = pd.read_csv(Path("../data/processed/y_train.csv")).to_numpy().ravel()
y_test = pd.read_csv(Path("../data/processed/y_test.csv")).to_numpy().ravel()

def train_rf() -> None:
    """Train a Random Forest Classifier."""
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(x_train, y_train)
    preds = rf_model.predict(x_test)
    logger.info(f"Random Forest Accuracy: {accuracy_score(y_test, preds)}")  # noqa: G004
    logger.info(classification_report(y_test, preds))

    with Path("../models/rf_model.pkl").open("wb") as f:
        pickle.dump(rf_model, f)
        logger.info("Random Forest model saved to ../models/rf_model.pkl")

class MLP(nn.Module):
    """Multi-Layer Perceptron for binary classification."""

    def __init__(self, input_size: int) -> None:
        """Initialise the MLP model."""
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Define the forward pass."""
        return self.layers(x)

def train_mlp() -> None:
    """Train a Multi-Layer Perceptron model."""
    x_train_tensor = torch.tensor(x_train.values, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)

    model = MLP(x_train.shape[1])
    criterion = nn.BCELoss()
    optimiser = optim.Adam(model.parameters(), lr=0.01)

    for _epoch in range(100):
        optimiser.zero_grad()
        outputs = model(x_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimiser.step()

    torch.save(model.state_dict(), "../models/mlp_model.pth")
    logger.info("MLP trained and saved to ../models/mlp_model.pth")

if __name__ == "__main__":
    logger.info("Training Random Forest model...")
    train_rf()
    logger.info("Training MLP model...")
    train_mlp()
