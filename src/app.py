"""Streamlit application for PyGol explainability analysis."""
import re
from pathlib import Path

import pandas as pd
import streamlit as st
from PyGol_Tabular.PyGol_Tabular import PyGolCounterfactual, PyGolMultiClassifier

SYMBOLIC_DATA_DIR = Path("../data/symbolic")
RULES_DIR = Path("../results/pygol")

st.set_page_config(layout="wide")

@st.cache_resource
def get_trained_model() -> tuple[PyGolMultiClassifier, pd.DataFrame]:
    """Load data and train PyGol Tabular model."""
    df = pd.read_csv(SYMBOLIC_DATA_DIR / "symbolic_diabetes.csv")
    x = df.drop(columns=["outcome"])
    y = df["outcome"]

    # exact_literals = true/false
    clf = PyGolMultiClassifier(
        binner="entropy",
        max_literals=2,
        n_bins=5,
        verbose=False,
    )
    clf.fit(x, y)

    # rules = clf._classifiers
    # print(rules)

    return clf, df

def run_tabular_explanation(patient_df: pd.DataFrame, full_train_df: pd.DataFrame) -> None:
    """Run PyGol Tabular counterfactual explanation for a given patient."""
    st.subheader("Counterfactual Analysis")

    cf_gen = PyGolCounterfactual(clf, full_train_df.drop(columns=["outcome"]))

    result = cf_gen.flip_example(
        patient_df,
        row_index=0,
        class_names={0: "Non-Diabetic", 1: "Diabetic"},
        verify=True,
    )

    # print(result.source_rules_fired)
    # print(result.target_rules_fired)

    if result.n_changes > 0:
        display_label = "Non-Diabetic" if result.target_class == 0 else "Diabetic"
        st.success(f"To change this prediction to **{display_label}**, the following changes are needed:")
        # st.write(result.print)

        for change in result.changes:
            # st.write(change)
            st.write(f"**{change['feature']}**: {change['from_value']} → {change['to_value']}")
    else:
        st.info("No counterfactuals found to change the prediction.")


def get_counterfactual_suggestions(patient_idx: int, rules_json: dict) -> None:
    """Retrieve and display counterfactual rules for a given patient index."""
    results = rules_json.get("analysis_results", {}).get(patient_idx, [])

    # If no rules are found, display a warning message
    if not results:
        st.warning("No counterfactual suggestions found for this patient.")
        return

    for rule in results:
        # Remove the "target(A):-"
        rule_content = rule.replace("target(A):-", "")

        # Use regex to extract feature names and their ranges
        pattern = r"(\w+)\(A,\w+\),inRange\(\w+,([\d\.]+--[\d\.]+)\)"
        matches = re.findall(pattern, rule_content)

        if matches:
            # Store suggestions in a list to display later
            suggestions = []

            # Format range (e.g. "0.5--1.0" -> "0.5 - 1.0")
            for feature, r_range in matches:
                clean_range = r_range.replace("--", " - ")
                suggestions.append(f"{feature} must be between: **({clean_range})**")

            # Display the suggestions in info box
            suggestion_text = "  \n".join(suggestions)
            st.info(suggestion_text)

        st.markdown("---")

# -----------------------------------------------------------------------------

clf, symbolic_data = get_trained_model()

st.title("PyGol Explainability Analysis")

patient_list = [f"Patient {i}" for i in symbolic_data.index]

select_patient_str = st.selectbox("Select a patient to analyse:", options=patient_list)
selected_idx = int(select_patient_str.split(" ")[1])

patient_row = symbolic_data.iloc[[selected_idx]].drop(columns=["outcome"])
actual_outcome = "Diabetic" if symbolic_data.iloc[selected_idx]["outcome"] == 1 else "Non-Diabetic"

col1, col2 = st.columns([1, 2])

with col1:
    st.write("### Patient Clinical Data")
    st.dataframe(patient_row.T.rename(columns={selected_idx: "Value"}))

with col2:
    run_tabular_explanation(patient_row, symbolic_data)
