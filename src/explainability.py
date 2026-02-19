"""Explainability module using DiCE and PyGol.

Generates counterfactual scenarios and symbolic logical rules for RF and MLP models.
"""

import json
import logging
import pickle
import warnings
from pathlib import Path

import dice_ml
import pandas as pd
import torch
from pandas.errors import PerformanceWarning
from PyGol import (
    bottom_clause_generation,
    evaluate_theory_prolog,
    prepare_examples,
    prepare_logic_rules,
    print_rules,
    pygol_learn,
    pygol_train_test_split,
    read_constants_meta_info,
)
from raiutils.exceptions import UserConfigValidationException
from torch import nn

from models import MLP, set_seed

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

MODELS_DIR = Path("../models")
PROCESSED_DATA_DIR = Path("../data/processed")
SYMBOLIC_DATA_DIR = Path("../data/symbolic")
DICE_DIR = Path("../results/dice")
RULES_DIR = Path("../results/pygol")

warnings.filterwarnings("ignore", category=PerformanceWarning)

def load_resources() -> tuple[object, MLP, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    """Load trained models, data (processed), and artifacts."""
    # 1. Load Data
    x_train = pd.read_csv(PROCESSED_DATA_DIR / "x_train.csv")
    y_train = pd.read_csv(PROCESSED_DATA_DIR / "y_train.csv")
    x_test = pd.read_csv(PROCESSED_DATA_DIR / "x_test.csv")
    y_test = pd.read_csv(PROCESSED_DATA_DIR / "y_test.csv")

    # 2. Load Artifacts
    with Path(PROCESSED_DATA_DIR / "preprocess_artifacts.pkl").open("rb") as f:
        artifacts = pickle.load(f)
    feature_names = artifacts["columns"]

    # 3. Load Random Forest
    with Path(MODELS_DIR / "rf_model.pkl").open("rb") as f:
        rf_model = pickle.load(f)

    # 4. Load Decision Tree
    with Path(MODELS_DIR / "dt_model.pkl").open("rb") as f:
        dt_model = pickle.load(f)

    # 5. Load MLP
    mlp_model = MLP(input_size=len(feature_names))
    mlp_model.load_state_dict(torch.load(MODELS_DIR / "mlp_model.pth", weights_only=True))
    mlp_model.eval()

    return rf_model, dt_model, mlp_model, x_train, y_train, x_test, y_test, feature_names


def load_symbolic_data() -> tuple[pd.DataFrame, pd.Series]:
    """Load data prepared for symbolic rule extraction."""
    x_train_symbolic = pd.read_csv(SYMBOLIC_DATA_DIR / "x_train_symbolic.csv")
    y_train_symbolic = pd.read_csv(SYMBOLIC_DATA_DIR / "y_train_symbolic.csv")["Outcome"].astype(int)
    return x_train_symbolic, y_train_symbolic


# -----------------------------------------------------------------------------
# DiCE
# -----------------------------------------------------------------------------
def make_permitted_range(
        x_train: pd.DataFrame,
        features: list[str],
        q_low: float = 0.01,
        q_high: float = 0.99,
    ) -> dict:
    """Create permitted ranges for DiCE continuous features based on training data quantiles."""
    permitted_ranges = {}
    quantiles = x_train[features].quantile([q_low, q_high])
    for f in features:
        permitted_ranges[f] = [float(quantiles.loc[q_low, f]), float(quantiles.loc[q_high, f])]
    return permitted_ranges


def run_dice(  # noqa: PLR0913
        model: object,
        x_train: pd.DataFrame,
        y_train: pd.DataFrame,
        x_test: pd.DataFrame,
        feature_names: list[str],
        patient_idx: int = 0,
        model_name: str = "Random Forest",
    ) -> None:
    """Generate DiCE counterfactuals for given model."""
    logger.info("Generating DiCE counterfactual explanations...")

    # Define continuous variables (Age/Pregnancies are discrete)
    continuous_vars = feature_names.copy()
    continuous_vars.remove("Pregnancies")
    continuous_vars.remove("Age")

    # Combine x_train and y_train for DiCE
    train_dataset = pd.concat([x_train, y_train], axis=1)
    d = dice_ml.Data(dataframe=train_dataset, continuous_features=continuous_vars, outcome_name="Outcome")

    # Define permitted ranges for changeable features
    actionable_features = ["Glucose", "BMI", "BloodPressure", "Insulin"]
    permitted_range = make_permitted_range(x_train, actionable_features, 0.05, 0.95)
    # logger.info(f"Permitted ranges for DiCE: {permitted_range}")

    query_instance = x_test.iloc[[patient_idx]]

    if model_name == "Multi-Layer Perceptron":
        # Wrap MLP to output probabilities for DiCE
        class DiceMLPWrapper(nn.Module):
            def __init__(self, base_model: nn.Module) -> None:
                super().__init__()
                self.base_model = base_model

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                logits = self.base_model(x).float()
                p1 = torch.sigmoid(logits)
                p0 = 1.0 - p1
                return torch.cat([p0, p1], dim=1)

        model.eval()
        wrapped_model = DiceMLPWrapper(model)
        m = dice_ml.Model(model=wrapped_model, backend="PYT")

        # Use genetic method for MLP, If fails, fallback to random
        methods = [
            ("genetic", {"proximity_weight": 0.5, "diversity_weight": 1.0}),
            ("random", {}),
        ]
    else:
        # Random Forest/Decision Tree
        m = dice_ml.Model(model=model, backend="sklearn")
        methods = [
            ("random", {}),
        ]

    # Try generating counterfactuals with different methods
    dice_exp = None
    used_method = None
    last_err: Exception | None = None

    for method, cf_params in methods:
        try:
            exp = dice_ml.Dice(d, m, method=method)
            dice_exp = exp.generate_counterfactuals(
                query_instance,
                total_CFs=5,
                desired_class="opposite",
                permitted_range=permitted_range,
                features_to_vary=actionable_features,
                **cf_params,
            )
            used_method = method
            break
        except UserConfigValidationException as e:
            last_err = e
            logger.warning(
                f"DiCE ({method}) could not generate CFs for patient {patient_idx} "
                f"using {model_name}. Retrying with next method...",
            )
        except Exception as e:  # noqa: BLE001
            last_err = e
            logger.warning(
                f"DiCE ({method}) failed for patient {patient_idx} using {model_name}: {e}. "
                f"Retrying with next method...",
            )

    if dice_exp is None or used_method is None:
        logger.error(
            f"All DiCE methods failed for patient {patient_idx} using {model_name}. "
            f"Last error: {last_err}",
        )
        return

    # Create output directory
    model_folder = model_name.replace(" ", "_").lower()
    patient_folder = f"patient_{patient_idx:03d}"
    output_dir = DICE_DIR / model_folder / patient_folder
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save counterfactuals to JSON
    json_obj = json.loads(dice_exp.to_json())
    json_obj["metadata"] = {
        "model_name": model_name,
        "patient_idx": patient_idx,
        "dice_method_used": used_method,
        "actionable_features": actionable_features,
        "permitted_range_quantiles": [0.05, 0.95],
    }

    with (output_dir / "dice_counterfactuals.json").open("w") as f:
        json.dump(json_obj, f, indent=2)

    logger.info(
        f"DiCE explanations saved to {output_dir / 'dice_counterfactuals.json'} "
        f"(method_used={used_method})",
    )

    dice_exp.visualize_as_dataframe()


# -----------------------------------------------------------------------------
# PyGol
# -----------------------------------------------------------------------------
def save_hypothesis(hypothesis: list[str], filename: str = "pygol_diabetes_rules.json") -> None:
    """Save PyGol hypothesis (rules) to a JSON file."""
    # Structure the data with metadata
    data_to_save = {
        "hypothesis": hypothesis,
    }

    with (RULES_DIR / filename).open("w") as f:
        json.dump(data_to_save, f, indent=2)

    logger.info(f"PyGol hypothesis saved to {RULES_DIR / filename}")


def run_pygol(
        x_train_symbolic: pd.DataFrame,
        y_train_symbolic: pd.Series,
        feature_names: list[str],
        target_name: str = "Outcome",
    ) -> list[str]:
    """Extract symbolic logical rules using PyGol on symbolic (unscaled, no-SMOTE) data."""
    logger.info("Generating PyGol symbolic logical rules...")

    # 1. Prepare data for PyGol
    data = pd.concat(
        [x_train_symbolic.reset_index(drop=True), y_train_symbolic.rename(target_name).reset_index(drop=True)],
        axis=1,
    )

    # 2. Generate background knowledge
    background = prepare_logic_rules(
        data,
        feature_names,
        meta_information="meta_data.info",
        default_div=5,
        conditions={},
    )

    # 3. Generate example files (pos_example.f and neg_example.n)
    examples = prepare_examples(data, target_name)

    # 4. Generate constant list from meta information
    const = read_constants_meta_info()

    # 5. Generate bottom clauses
    P, N = bottom_clause_generation(
        file="BK.pl",
        constant_set=const,
        container="dict",
        positive_example="pos_example.f",
        negative_example="neg_example.n",
    )

    # 6. Split into train/test sets for PyGol
    Train_P, Test_P, Train_N, Test_N = pygol_train_test_split(
        test_size=0.25,
        positive_file_dictionary=P,
        negative_file_dictionary=N,
    )

    # 7. Perform Learning
    model = pygol_learn(
        Train_P,
        Train_N,
        max_literals=4,
        key_size=1,
        min_pos=2,
        max_neg=0,
    )

    # print_rules(model.hypothesis, meta_information = "meta_data.info")

    if model.hypothesis:
        save_hypothesis(model.hypothesis)
    else:
        logger.warning("No hypothesis (rules) generated by PyGol.")

    return evaluate_theory_prolog(model.hypothesis, "BK.pl", Test_P, Test_N)


if __name__ == "__main__":
    set_seed(42)

    # Load resources
    rf_model, dt_model, mlp_model, x_train, y_train, x_test, y_test, feature_names = load_resources()

    # 1. Generate DiCE counterfactuals for both models
    for i in range(3):
        run_dice(rf_model, x_train, y_train, x_test, feature_names, model_name="Random Forest", patient_idx=i)
        run_dice(dt_model, x_train, y_train, x_test, feature_names, model_name="Decision Tree", patient_idx=i)
        run_dice(mlp_model, x_train, y_train, x_test, feature_names, model_name="Multi-Layer Perceptron", patient_idx=i)

    # Load symbolic data
    x_train_symbolic, y_train_symbolic = load_symbolic_data()

    # 2. Generate PyGol symbolic logical rules
    rules = run_pygol(x_train_symbolic, y_train_symbolic, feature_names)

    logger.info("Explainability tasks completed.")
