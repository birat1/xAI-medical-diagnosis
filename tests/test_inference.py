"""Test inference of trained models."""
import logging
import pickle
import sys
from pathlib import Path

import pandas as pd
import torch

src_path = str(Path(__file__).parent.parent / "src")
if src_path not in sys.path:
    sys.path.append(src_path)

from models import MLP  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

RF_THRESHOLD = 0.3775
MLP_THRESHOLD = 0.4173

try:
    sample = pd.read_csv("../data/processed/x_test.csv").iloc[[0]]
    y_test_sample = pd.read_csv("../data/processed/y_test.csv").iloc[[0]].to_numpy().ravel()[0]
except FileNotFoundError as e:
    logger.exception(f"Test data not found: {e}")
    sys.exit(1)

# Test Random Forest
with Path("../models/rf_model.pkl").open("rb") as f:
    rf_model = pickle.load(f)
    rf_pred = 1 if rf_model.predict_proba(sample)[0][1] > RF_THRESHOLD else 0
    rf_prob = rf_model.predict_proba(sample)[0][1]
    logger.info(f"RF Prediction: {rf_pred} (Probability: {rf_prob:.4f})")

# Test MLP
model = MLP(sample.shape[1])
model.load_state_dict(torch.load("../models/mlp_model.pth", weights_only=True))
model.eval()

with torch.no_grad():
    sample_tensor = torch.tensor(sample.values, dtype=torch.float32)
    logits = model(sample_tensor)
    prob = torch.sigmoid(logits).item()
    mlp_pred = 1 if prob > MLP_THRESHOLD else 0
    logger.info(f"MLP Prediction: {mlp_pred} (Probability: {prob:.4f})")

logger.info(f"Actual Label: {y_test_sample}")
