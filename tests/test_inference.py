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

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
)

sample = pd.read_csv("../data/processed/x_test.csv").iloc[[0]]

# Test Random Forest
with Path("../models/rf_model.pkl").open("rb") as f:
    rf_model = pickle.load(f)
    logger.info(f"RF Prediction: {rf_model.predict(sample)}")  # noqa: G004

# Test MLP
model = MLP(sample.shape[1])
model.load_state_dict(torch.load("../models/mlp_model.pth"))
model.eval()
with torch.no_grad():
    sample_tensor = torch.tensor(sample.values, dtype=torch.float32)
    prediction = model(sample_tensor)
    logger.info(f"MLP Prediction: {prediction.item():.4f}")  # noqa: G004
