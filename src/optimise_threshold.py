"""Optimise thresholds for classification models to maximise F1-Score."""
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import average_precision_score, precision_recall_curve

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = Path("../results")
MODELS_DIR = Path("../models")
METRICS_DIR = Path("../results/metrics")
METRICS_DIR.mkdir(parents=True, exist_ok=True)

def find_optimal_threshold(
        y_true: np.ndarray,
        y_probs: np.ndarray,
        model_name: str = "Model",
    ) -> dict[str, float | list[float]]:
    """Find threshold that maximises the F1-Score."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_probs)

    precision = precision[:-1]
    recall = recall[:-1]

    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)

    best_idx = int(np.argmax(f1_scores))
    best_threshold = float(thresholds[best_idx])
    best_f1 = float(f1_scores[best_idx])
    best_precision = float(precision[best_idx])
    best_recall = float(recall[best_idx])
    auprc = float(average_precision_score(y_true, y_probs))


    logger.info(f"\n--- {model_name} Threshold Optimisation ---")
    logger.info(f"AUPRC: {auprc:.4f}")
    logger.info(f"Optimal Threshold: {best_threshold:.4f}")
    logger.info(f"Best F1: {best_f1:.4f}")
    logger.info(f"Precision @ best threshold: {best_precision:.4f}")
    logger.info(f"Recall @ best threshold: {best_recall:.4f}")

    return {
        "threshold": best_threshold,
        "best_f1": best_f1,
        "precision": best_precision,
        "recall": best_recall,
        "auprc": auprc,
        "thresholds": thresholds.tolist(),
        "f1_scores": f1_scores.tolist(),
    }

def plot_f1_threshold_curves(results: dict[str, dict[str, float | list[float]]]) -> None:
    """Plot F1-Threshold curves for all models."""
    plt.figure(figsize=(8, 6))

    for model_name, result in results.items():
        thresholds = np.array(result["thresholds"], dtype=float)
        f1_scores = np.array(result["f1_scores"], dtype=float)
        best_threshold = float(result["threshold"])

        plt.plot(thresholds, f1_scores, label=f"{model_name} (best={best_threshold:.3f})")

    plt.axvline(0.5, color="red", linestyle="--", label="Default 0.5")
    plt.xlabel("Threshold")
    plt.ylabel("F1-Score")
    plt.title("Threshold Optimisation")
    plt.legend()
    plt.grid(visible=True, linestyle="--", alpha=0.5)

    out_dir = METRICS_DIR / "f1_threshold_optimisation.png"
    plt.savefig(out_dir, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"F1-Threshold Plot saved to {out_dir}")

def plot_calibration_curves(y_true: np.ndarray, probs_dict: dict[str, np.ndarray]) -> None:
    """Plot calibration curves for all models."""
    plt.figure(figsize=(8, 6))

    for model_name, probs in probs_dict.items():
        prob_true, prob_pred = calibration_curve(y_true, probs, n_bins=10, pos_label=1)
        plt.plot(prob_pred, prob_true, marker="o", label=model_name)

    plt.plot([0, 1], [0, 1], linestyle="--", label="Perfectly Calibrated", color="gray")
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Fraction of Positives")
    plt.title("Calibration Curve")
    plt.legend()
    plt.grid(visible=True, linestyle="--", alpha=0.5)

    out_dir = METRICS_DIR / "calibration_curve.png"
    plt.savefig(out_dir, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"Calibration Curve saved to {out_dir}")

def save_thresholds(results: dict[str, dict[str, float | list[float]]]) -> None:
    """Save tuned thresholds and metrics to JSON."""
    thresholds = {
        "dt_threshold": float(results["dt"]["threshold"]),
        "rf_threshold": float(results["rf"]["threshold"]),
        "mlp_threshold": float(results["mlp"]["threshold"]),
    }

    out_dir = METRICS_DIR / "thresholds.json"
    with out_dir.open("w") as f:
        json.dump(thresholds, f, indent=2)

    logger.info(f"Tuned thresholds saved to {out_dir}")

if __name__ == "__main__":
    y_dev = np.load(MODELS_DIR / "y_dev.npy")

    dt_oof_probs = np.load(MODELS_DIR / "dt_oof_probs.npy")
    rf_oof_probs = np.load(MODELS_DIR / "rf_oof_probs.npy")
    mlp_oof_probs = np.load(MODELS_DIR / "mlp_oof_probs.npy")

    probs_dict = {
        "DT": dt_oof_probs,
        "RF": rf_oof_probs,
        "MLP": mlp_oof_probs,
    }

    results = {
        "dt": find_optimal_threshold(y_dev, dt_oof_probs, model_name="Decision Tree"),
        "rf": find_optimal_threshold(y_dev, rf_oof_probs, model_name="Random Forest"),
        "mlp": find_optimal_threshold(y_dev, mlp_oof_probs, model_name="Multi-Layer Perceptron"),
    }

    save_thresholds(results)
    plot_f1_threshold_curves(results)
    plot_calibration_curves(y_dev, probs_dict)

    logger.info("Threshold optimisation completed successfully.")
