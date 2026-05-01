"""Streamlit application for PyGol explainability analysis."""
import json
import pickle
from pathlib import Path
from unittest import result

import numpy as np
import pandas as pd
import streamlit as st
from PyGol_Tabular.PyGol_Tabular import PyGolCounterfactual, PyGolMultiClassifier

# Paths
PROCESSED_DATA_DIR = Path("../data/processed")
SYMBOLIC_DATA_DIR = Path("../data/symbolic")
DICE_DIR = Path("../results/dice")
RULES_DIR = Path("../results/pygol")

st.set_page_config(layout="wide")

### Formatting functions
def _model_name_to_folder(model_name: str) -> str:
    """Convert model name to folder name format."""
    return model_name.replace(" ", "_").lower()

def _format_label(label: int) -> str:
    """Format numeric label into string."""
    return "Non-Diabetic" if label == 0 else "Diabetic"

def _format_feature_name(feature: str) -> str:
    """Format feature name for display."""
    readable_names = {
        "pregnancies": "Pregnancies",
        "glucose": "Glucose",
        "bloodpressure": "Blood Pressure",
        "skinthickness": "Skin Thickness",
        "insulin": "Insulin",
        "bmi": "BMI",
        "diabetespedigreefunction": "Diabetes Pedigree Function",
        "age": "Age",
    }

    return readable_names.get(feature, feature.replace("_", " ").title())

def _format_rule_condition(feature: str, value: str) -> str:
    """Format a single PyGol rule condition for display."""
    feature_name = _format_feature_name(feature)
    value = str(value)

    if value.startswith("[") and value.endswith("]") and "," in value:
        lower, upper = value.strip("[]").split(",", maxsplit=1)
        return f"{feature_name} is between {lower} and {upper}"

    return f"{feature_name} equals {value}"

def _format_counterfactual_change(change: dict) -> str:
    """Format counterfactual change for display."""
    feature = _format_feature_name(change["feature"])
    from_value = change["from_value"]
    to_value = change["to_value"]

    try:
        from_num = float(from_value)
        to_num = float(to_value)

        if to_num > from_num:
            direction = "increase"
            difference = to_num - from_num
        elif to_num < from_num:
            direction = "decrease"
            difference = from_num - to_num
        else:
            return f"**{feature}** remains at {to_value}."

        return (  # noqa: TRY300
            f"**{feature}** should {direction} from {from_value} to {to_value} by approximately {difference:.2f}."
        )
    except (ValueError, TypeError):
        return (
            f"**{feature}** should change from {from_value} to {to_value}."
        )

def _round_clinical_df(df: pd.DataFrame) -> pd.DataFrame:
    """Round clinical features in DataFrame for better display."""
    return df.round({
        "pregnancies": 0,
        "glucose": 0,
        "bloodpressure": 0,
        "skinthickness": 1,
        "insulin": 1,
        "bmi": 1,
        "diabetespedigreefunction": 3,
        "age": 0,
    })

### Helpers
def _rules_to_conditions(rule: dict) -> list[str]:
    """Convert a PyGol rule into human readable conditions."""
    return [
        _format_rule_condition(feature, value)
        for feature, value in rule.items()
        if feature != "__label__"
    ]

def _get_rule_label(rules: list[dict]) -> str:
    """Get readable label from first fired rule."""
    if not rules:
        return "Unknown"
    return _format_label(rules[0].get("__label__", "Unknown"))

def _display_rule_cards(rules: list[dict], title: str) -> None:
    """Display PyGol rules as IF-THEN cards."""
    st.markdown(f"{title}")

    if not rules:
        st.info(f"No {title.lower()} were fired.")
        return

    predicted_label = _get_rule_label(rules)
    st.caption(f"Prediction: **{predicted_label}**")

    for idx, rule in enumerate(rules, start=1):
        conditions = _rules_to_conditions(rule)

        with st.container(border=True):
            st.markdown(f"**Rule {idx}**")

            st.markdown("**IF**")
            for condition in conditions:
                st.markdown(f"- {condition}")

def display_pygol_rules(source_rules: list[dict], target_rules: list[dict]) -> None:
    """Display PyGol source and target rules in a human readable format."""
    with st.expander("Show PyGol rules behind the prediction", expanded=False):
        _display_rule_cards(source_rules, "Source Rules (Original Prediction)")

        st.markdown("---")

        _display_rule_cards(target_rules, "Target Rules (Counterfactual Prediction)")

### Data Loading
@st.cache_resource
def load_preprocessing_artifacts() -> dict:
    """Load preprocessing artifacts needed to inverse-transform DiCE outputs."""
    artifacts_dir = PROCESSED_DATA_DIR / "preprocess_artifacts.pkl"
    if not artifacts_dir.exists():
        raise FileNotFoundError(f"Preprocessing artifacts not found at {artifacts_dir}")  # noqa: EM102, TRY003

    with artifacts_dir.open("rb") as f:
        return pickle.load(f)

@st.cache_resource
def get_trained_model() -> tuple[PyGolMultiClassifier, pd.DataFrame, pd.DataFrame]:
    """Load symbolic train/test data and train PyGol Tabular model."""
    symbolic_x_train = pd.read_csv(SYMBOLIC_DATA_DIR / "symbolic_x_train.csv")
    symbolic_y_train = pd.read_csv(SYMBOLIC_DATA_DIR / "symbolic_y_train.csv")
    symbolic_x_test = pd.read_csv(SYMBOLIC_DATA_DIR / "symbolic_x_test.csv")
    symbolic_y_test = pd.read_csv(SYMBOLIC_DATA_DIR / "symbolic_y_test.csv")

    # exact_literals = true/false
    clf = PyGolMultiClassifier(
        binner="entropy",
        max_literals=2,
        n_bins=5,
        verbose=False,
    )
    clf.fit(symbolic_x_train, symbolic_y_train["outcome"])

    symbolic_train_data = pd.concat([symbolic_x_train, symbolic_y_train], axis=1)
    symbolic_test_data = pd.concat([symbolic_x_test, symbolic_y_test], axis=1)

    # rules = clf._classifiers
    # print(rules)

    return clf, symbolic_train_data, symbolic_test_data

### PyGol
def run_tabular_explanation(clf: PyGolMultiClassifier, patient_df: pd.DataFrame, full_train_df: pd.DataFrame) -> None:
    """Run PyGol Tabular counterfactual explanation for a given patient."""
    st.subheader("PyGol Counterfactuals")

    cf_gen = PyGolCounterfactual(clf, full_train_df.drop(columns=["outcome"]))

    result = cf_gen.flip_example(
        patient_df,
        row_index=0,
        class_names={0: "Non-Diabetic", 1: "Diabetic"},
        verify=True,
    )

    if result.n_changes > 0:
        display_label = "Non-Diabetic" if result.target_class == 0 else "Diabetic"
        st.success(f"To change this prediction to **{display_label}**, the following changes are needed:")
        # st.write(result)

        for change in result.changes:
            # st.write(change)
            st.write(_format_counterfactual_change(change))

        st.caption(f"Features changed: {result.n_changes}")
        st.caption(f"Target rules fired: {len(result.target_rules_fired)}")
        display_pygol_rules(source_rules=result.source_rules_fired, target_rules=result.target_rules_fired)
    else:
        st.info("No counterfactuals found to change the prediction.")

### DiCE
def _load_dice_json(patient_idx: int, model_name: str) -> dict | None:
    """Load DiCE JSON file for a given patient and model."""
    model_folder = _model_name_to_folder(model_name)
    patient_folder = f"patient_{patient_idx:03d}"

    dice_dir = DICE_DIR / model_folder / patient_folder / "dice_counterfactuals.json"
    if not dice_dir.exists():
        st.warning(f"No DiCE file found at: {dice_dir}")
        return None

    with dice_dir.open("r") as f:
        return json.load(f)

def load_dice_original_patient(patient_idx: int, model_name: str) -> pd.DataFrame | None:
    """Load original DiCE patient and convert from scaled values to clinical units."""
    dice_json = _load_dice_json(patient_idx, model_name)
    if dice_json is None:
        return None

    artifacts = load_preprocessing_artifacts()
    scaler = artifacts["scaler"]
    target_cols = artifacts["columns"]

    feature_names = dice_json.get("feature_names")
    idx_map = [feature_names.index(col) for col in target_cols]

    test_data_row = dice_json["test_data"][0][0]
    orig_outcome = test_data_row[-1]

    x_scaled = np.array(test_data_row, dtype=float)[idx_map]
    x_clinical = scaler.inverse_transform(x_scaled.reshape(1, -1)).ravel()

    orig_df = pd.DataFrame([x_clinical], columns=target_cols)
    orig_df["Outcome"] = orig_outcome

    return _round_clinical_df(orig_df)

def load_dice_counterfactuals(patient_idx: int, model_name: str) -> pd.DataFrame | None:
    """Load DiCE counterfactuals and convert scaled values to clinical."""
    dice_json = _load_dice_json(patient_idx, model_name)
    if dice_json is None:
        return None

    artifacts = load_preprocessing_artifacts()
    scaler = artifacts["scaler"]
    target_cols = artifacts["columns"]

    feature_names = dice_json.get("feature_names")
    idx_map = [feature_names.index(col) for col in target_cols]

    cfs_list = dice_json["cfs_list"][0]
    if not cfs_list:
        st.warning("DiCE counterfactuals list is empty.")
        return None

    cf_outcomes = [cf[-1] for cf in cfs_list]

    cfs_scaled = np.array(
        [cf[:len(feature_names)] for cf in cfs_list],
        dtype=float,
    )[:, idx_map]

    cfs_clinical = scaler.inverse_transform(cfs_scaled)

    cf_df = pd.DataFrame(cfs_clinical, columns=target_cols)
    cf_df.insert(0, "Counterfactual ID", [f"CF {i+1}" for i in range(len(cf_df))])
    cf_df["Outcome"] = cf_outcomes

    return _round_clinical_df(cf_df)

def calculate_dice_summary(orig_df: pd.DataFrame, cf_df: pd.DataFrame) -> dict:
    """Calculate changes between original and counterfactual DataFrames for summary."""
    feature_cols = [
        col for col in cf_df.columns
        if col not in ["Counterfactual ID", "Outcome"]
    ]

    orig_values = orig_df[feature_cols].iloc[0]
    cf_values = cf_df[feature_cols]

    delta = cf_values - orig_values

    changed = delta.abs() > 1e-6
    changes_per_cf = changed.sum(axis=1)

    return {
        "num_cfs": len(cf_df),
        "min_changes": changes_per_cf.min(),
        "max_changes": changes_per_cf.max(),
        "changes_per_cf": changes_per_cf,
    }

def display_dice_counterfactuals(patient_idx: int) -> None:
    """Display DiCE counterfactuals for the selected patient."""
    st.subheader("DiCE Counterfactuals")

    model_name = st.selectbox(
        "Select model for DiCE results:",
        options=[
            "Random Forest",
            "Decision Tree",
            "Multi-Layer Perceptron",
        ],
    )

    orig_df = load_dice_original_patient(patient_idx=patient_idx, model_name=model_name)
    dice_df = load_dice_counterfactuals(patient_idx=patient_idx, model_name=model_name)

    if dice_df is None:
        st.info(
            "Run `generate_counterfactuals.py` first to create a saved DiCE JSON file.",
        )
        return

    if orig_df is not None:
        dice_summary = calculate_dice_summary(orig_df, dice_df)

        st.caption(f"Counterfactuals generated: {dice_summary['num_cfs']}")
        st.caption(f"Minimum feature changes: {dice_summary['min_changes']}")
        st.caption(f"Max feature changes: {dice_summary['max_changes']}")

    st.write(f"Showing counterfactuals for **{model_name}**:")
    st.dataframe(dice_df, width="stretch")


# Streamlit App
clf, symbolic_train_data, symbolic_test_data = get_trained_model()

st.title("Counterfactual Analysis")
st.caption("PyGol uses symbolic version of the test set. DiCE uses the processed version of the same test patients.")

patient_list = [f"Patient {i}" for i in symbolic_test_data.index]

select_patient_str = st.selectbox("Select a patient to analyse:", options=patient_list)
selected_idx = int(select_patient_str.split(" ")[1])

patient_row = symbolic_test_data.iloc[[selected_idx]].drop(columns=["outcome"])
actual_outcome = _format_label(symbolic_test_data.iloc[selected_idx]["outcome"])

col1, col2 = st.columns([1, 2])

with col1:
    st.write("### Patient Clinical Data")
    st.dataframe(patient_row.T.rename(columns={selected_idx: "Value"}))

with col2:
    run_tabular_explanation(clf=clf, patient_df=patient_row, full_train_df=symbolic_train_data)

st.markdown("---")

if selected_idx <= 4: # Only show DiCE results for first 5 patients (since we only generated for those)
    display_dice_counterfactuals(selected_idx)
else:
    st.info("DiCE counterfactuals are currently only available for the first 5 patients.")
