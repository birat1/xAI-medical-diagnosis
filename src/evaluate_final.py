"""Final evaluation on the held-out test set."""

import json
import logging
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)

from models import MLP

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path("../data/processed")
MODELS_DIR = Path("../models")
METRICS_DIR = Path("../results/metrics")
METRICS_DIR.mkdir(parents=True, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_data() -> tuple[pd.DataFrame, np.ndarray]:
    """Load test data."""
    x_test = pd.read_csv(DATA_DIR / "x_test.csv")
    y_test = pd.read_csv(DATA_DIR / "y_test.csv").to_numpy().ravel()

    return x_test, y_test

def load_preprocessor() -> dict[str, object]:
    """Load fitted final preprocessor artifacts."""
    with (MODELS_DIR / "preprocessor.pkl").open("rb") as f:
        artifacts = pickle.load(f)

    return artifacts

def transform_data(x_test: pd.DataFrame, artifacts: dict[str, object]) -> np.ndarray:
    """Transform test data with saved final preprocessor."""
    imputer = artifacts["imputer"]
    scaler = artifacts["scaler"]

    x_test_imputed = imputer.transform(x_test)
    x_test_scaled = scaler.transform(x_test_imputed)

    return x_test_scaled

def load_rf_model() -> object:
    """Load saved final Random Forest model."""
    with (MODELS_DIR / "rf_model.pkl").open("rb") as f:
        model = pickle.load(f)

    return model

def load_dt_model() -> object:
    """Load saved final Decision Tree model."""
    with (MODELS_DIR / "dt_model.pkl").open("rb") as f:
        model = pickle.load(f)

    return model

def load_mlp_model(input_size: int) -> MLP:
    """Load saved final MLP model."""
    model = MLP(input_size).to(device)
    state_dict = torch.load(MODELS_DIR / "mlp_model.pth", map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    return model

def load_thresholds() -> dict[str, float]:
    """Load tuned thresholds."""
    with (METRICS_DIR / "thresholds.json").open("r") as f:
        thresholds = json.load(f)

    return {
        "rf": float(thresholds["rf_threshold"]),
        "dt": float(thresholds["dt_threshold"]),
        "mlp": float(thresholds["mlp_threshold"]),
    }

def predict_probs(model: object, x_scaled: np.ndarray, is_pytorch: bool) -> np.ndarray:
    """Predict probabilities."""
    if is_pytorch:
        model.eval()
        with torch.no_grad():
            logits = model(torch.tensor(x_scaled, dtype=torch.float32).to(device))
            return torch.sigmoid(logits).cpu().numpy().ravel()

    return model.predict_proba(x_scaled)[:, 1]

def evaluate_at_threshold(
    y_true: np.ndarray,
    probs: np.ndarray,
    threshold: float,
) -> dict[str, object]:
    """Evaluate predictions at a given threshold."""
    preds = (probs >= threshold).astype(int)

    return {
        "threshold": float(threshold),
        "auprc": float(average_precision_score(y_true, probs)),
        "f1": float(f1_score(y_true, preds)),
        "precision": float(precision_score(y_true, preds, zero_division=0)),
        "recall": float(recall_score(y_true, preds, zero_division=0)),
    }

def save_eval_results(results: dict[str, object], filename: str) -> None:
    """Save evaluation results to JSON."""
    out_path = METRICS_DIR / filename
    with out_path.open("w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Saved evaluation results to {out_path}")

def plot_precision_recall_curves(
    y_test: np.ndarray,
    probs_dict: dict[str, np.ndarray],
) -> None:
    """Plot precision-recall curves for all final models."""
    plt.figure(figsize=(8, 6))

    for model_name, probs in probs_dict.items():
        precision, recall, _ = precision_recall_curve(y_test, probs)
        auprc = average_precision_score(y_test, probs)
        plt.plot(recall, precision, linewidth=1.8, label=f"{model_name} (AUPRC={auprc:.3f})")

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curves on Test Set")
    plt.legend()
    plt.grid(visible=True, linestyle="--", alpha=0.5)

    out_path = METRICS_DIR / "test_pr_curves.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved Precision-Recall curves to {out_path}")

if __name__ == "__main__":
    """Run final test-set evaluation."""
    x_test, y_test = load_data()
    artifacts = load_preprocessor()
    x_test_scaled = transform_data(x_test, artifacts)

    rf_model = load_rf_model()
    dt_model = load_dt_model()
    mlp_model = load_mlp_model(input_size=x_test.shape[1])

    thresholds = load_thresholds()

    rf_probs = predict_probs(rf_model, x_test_scaled, is_pytorch=False)
    dt_probs = predict_probs(dt_model, x_test_scaled, is_pytorch=False)
    mlp_probs = predict_probs(mlp_model, x_test_scaled, is_pytorch=True)

    probs_dict = {
        "Random Forest": rf_probs,
        "Decision Tree": dt_probs,
        "MLP": mlp_probs,
    }

    # Default threshold evaluation
    default_results = {
        "rf": evaluate_at_threshold(y_test, rf_probs, threshold=0.5),
        "dt": evaluate_at_threshold(y_test, dt_probs, threshold=0.5),
        "mlp": evaluate_at_threshold(y_test, mlp_probs, threshold=0.5),
    }

    # Tuned threshold evaluation
    tuned_results = {
        "rf": evaluate_at_threshold(y_test, rf_probs, threshold=thresholds["rf"]),
        "dt": evaluate_at_threshold(y_test, dt_probs, threshold=thresholds["dt"]),
        "mlp": evaluate_at_threshold(y_test, mlp_probs, threshold=thresholds["mlp"]),
    }

    combined_results = {}
    for model_name in ["rf", "dt", "mlp"]:
        combined_results[model_name] = {
            "default": default_results[model_name],
            "tuned": tuned_results[model_name],
        }

    save_eval_results(combined_results, "test_results.json")

    plot_precision_recall_curves(y_test, probs_dict)

    logger.info("\n=== Final Test Results (Default Threshold = 0.5) ===")
    for model_name, result in default_results.items():
        logger.info(
            f"{model_name.upper()} | "
            f"AUPRC: {result['auprc']:.4f} | "
            f"F1: {result['f1']:.4f} | "
            f"Precision: {result['precision']:.4f} | "
            f"Recall: {result['recall']:.4f}",
        )

    logger.info("\n=== Final Test Results (Tuned Thresholds) ===")
    for model_name, result in tuned_results.items():
        logger.info(
            f"{model_name.upper()} | "
            f"Threshold: {result['threshold']:.4f} | "
            f"AUPRC: {result['auprc']:.4f} | "
            f"F1: {result['f1']:.4f} | "
            f"Precision: {result['precision']:.4f} | "
            f"Recall: {result['recall']:.4f}",
        )

    logger.info("Final evaluation completed successfully.")
