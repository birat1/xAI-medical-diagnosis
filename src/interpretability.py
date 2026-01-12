"""Interpretability analysis using SHAP and LIME for diabetes prediction."""
import logging
import pickle
from pathlib import Path

import lime
import numpy as np
import pandas as pd
import shap
import torch
from matplotlib import pyplot as plt

from models import MLP

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

MODELS_DIR = Path("../models")
DATA_DIR = Path("../data/processed")
SHAP_DIR = Path("../results/shap")
LIME_DIR = Path("../results/lime")

def load_resources() -> tuple[object, MLP, pd.DataFrame, pd.DataFrame, list[str]]:
    """Load trained models, data, and artifacts."""
    # 1. Load Data
    x_train = pd.read_csv(DATA_DIR / "x_train.csv")
    x_test = pd.read_csv(DATA_DIR / "x_test.csv")

    # 2. Load Artifacts
    with Path(DATA_DIR / "preprocess_artifacts.pkl").open("rb") as f:
        artifacts = pickle.load(f)
    feature_names = artifacts["columns"]

    # 3. Load Random Forest
    with Path(MODELS_DIR / "rf_model.pkl").open("rb") as f:
        rf_model = pickle.load(f)

    # 4. Load MLP
    mlp_model = MLP(input_size=len(feature_names))
    mlp_model.load_state_dict(torch.load(MODELS_DIR / "mlp_model.pth", weights_only=True))
    mlp_model.eval()

    return rf_model, mlp_model, x_train, x_test, feature_names

def run_shap(model: object, x_test: pd.DataFrame, feature_names: list[str], name: str ="Random Forest") -> None:
    """Generate SHAP global feature importance plots."""
    logger.info(f"Generating SHAP explanations for {name}...")

    # SHAP for MLP
    if isinstance(model, torch.nn.Module):
        # Limit to first 100 samples for SHAP
        plot_data = x_test.iloc[:100]
        background = torch.tensor(x_test.to_numpy()[:100], dtype=torch.float32)
        explainer = shap.DeepExplainer(model, background)

        test_tensor = torch.tensor(plot_data.to_numpy(), dtype=torch.float32)
        shap_values = explainer.shap_values(test_tensor)

        shap_values_to_plot = shap_values[:, :, 0] if len(shap_values.shape) == 3 else shap_values
    # SHAP for Random Forest
    else:
        # Limit to first 100 samples for SHAP
        plot_data = x_test.iloc[:100]
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(plot_data)

        shap_values_to_plot = shap_values[1] if isinstance(shap_values, list) else shap_values[:, :, 1]

    # Global summary plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values_to_plot, plot_data, feature_names=feature_names, show=False)
    plt.title(f"{name}")
    plt.savefig(SHAP_DIR / f"shap_summary_{name.lower().replace(' ', '_')}.png", bbox_inches="tight", dpi=300)
    plt.close()

def run_lime(
        model: object,
        x_train: pd.DataFrame,
        x_test: pd.DataFrame,
        feature_names: list[str],
        name: str ="Random Forest",
    ) -> None:
    """Generate LIME explanations for a specific patient."""
    logger.info(f"Generating LIME explanations for {name}...")

    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=x_train.to_numpy(),
        feature_names=feature_names,
        class_names=["No Diabetes", "Diabetes"],
        mode="classification",
    )

    # If model is MLP, define a prediction function
    if isinstance(model, torch.nn.Module):
        def predict_fn(x: np.ndarray) -> np.ndarray:
            model.eval()
            with torch.no_grad():
                logits = model(torch.tensor(x, dtype=torch.float32))
                probs = torch.sigmoid(logits).numpy()
                return np.hstack((1 - probs, probs))
    # If model is Random Forest
    else:
        def predict_fn(x: np.ndarray) -> np.ndarray:
            x_df = pd.DataFrame(x, columns=feature_names)
            return model.predict_proba(x_df)

    # Explain the first instance in the test set
    idx = 0
    exp = explainer.explain_instance(x_test.to_numpy()[idx], predict_fn, num_features=len(feature_names))

    # Save as HTML
    output_path = LIME_DIR / f"lime_report_{name.lower().replace(' ', '_')}.html"
    exp.save_to_file(output_path)

    # Save as figure
    fig = exp.as_pyplot_figure()
    plt.figure(figsize=(10, 6))
    plt.title(f"LIME Explanation for {name.lower().replace(' ', '_')} - Instance {idx}")
    fig.savefig(LIME_DIR / f"lime_plot_{name.lower().replace(' ', '_')}.png", bbox_inches="tight", dpi=300)
    plt.close(fig)

if __name__ == "__main__":
    rf, mlp, x_train, x_test, features = load_resources()

    # 1. Interpret Random Forest
    run_shap(rf, x_test, features, name="Random Forest")
    run_lime(rf, x_train, x_test, features, name="Random Forest")

    # 2. Interpret MLP
    run_shap(mlp, x_test, features, name="Multi-Layer Perceptron")
    run_lime(mlp, x_train, x_test, features, name="Multi-Layer Perceptron")

    logger.info("Interpretability analysis completed.")
