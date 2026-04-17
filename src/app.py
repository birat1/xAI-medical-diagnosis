"""Streamlit application for PyGol explainability analysis."""
import json
import re
from pathlib import Path

import pandas as pd
import streamlit as st

SYMBOLIC_DATA_DIR = Path("../data/symbolic")
RULES_DIR = Path("../results/pygol")

st.set_page_config(layout="wide")

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

symbolic_data = pd.read_csv(SYMBOLIC_DATA_DIR / "symbolic_diabetes.csv")
with Path(RULES_DIR / "pygol_diabetes_rules.json").open("r") as f:
    rules_data = json.load(f)

# e.g. target(e_115)
patient_idx = list(rules_data["analysis_results"].keys())
# e.g. target(e_115) -> Patient 115
display_labels = {k: f"Patient {k.split('(e_')[1].split(')')[0]}" for k in patient_idx}

st.title("PyGol Explainability Analysis")
col1, col2 = st.columns([1, 2])

with col1:
    selected_patient = st.selectbox(
        "Select a patient to analyse:",
        options=patient_idx,
        format_func=lambda x: display_labels[x],
    )

    selected_idx = int(selected_patient.split("(e_")[1].split(")")[0])
    patient_row = symbolic_data.iloc[selected_idx].to_dict()

    st.write("**Patient Summary:**")
    st.json(patient_row)

with col2:
    st.subheader("Counterfactual Suggestions:")
    st.caption("(changes needed to flip prediction from non-diabetic to diabetic)")
    get_counterfactual_suggestions(selected_patient, rules_data)
