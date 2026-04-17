"""Streamlit application for PyGol explainability analysis."""
import json
import re
from pathlib import Path

import pandas as pd
import streamlit as st

RULES_DIR = Path("../results/pygol")
METADATA_FILE = Path("meta_data.info")

def parse_meta_data() -> dict:
    """Parse meta data from meta_data.info file."""
    metadata = {}
    curr_feature = None

    with (METADATA_FILE).open("r") as f:
        for line in f:
            line = line.strip()
            if "Feature Name :" in line:
                curr_feature = line.split(":")[1].split("(")[0].strip().lower()
                metadata[curr_feature] = {}
            elif curr_feature and re.match(r"^g\d", line):
                parts = line.split("\t")
                bin_id = parts[0].strip()
                val_range = parts[1].strip()
                metadata[curr_feature][bin_id] = val_range

    return metadata

@st.cache_data
def load_resources() -> tuple[dict, dict]:
    """Load meta data and PyGol rules from files."""
    metadata = parse_meta_data()

    with (RULES_DIR / "pygol_diabetes_rules.json").open("r") as f:
        rules = json.load(f)

    return metadata, rules

def get_rules(patient_data, hypothesis):
    boundary_conditions = []
    for rule in hypothesis:
        # e.g. rule: "target(A):-age(A,g4),bmi(A,g0),insulin(A,g1)"  # noqa: ERA001

        # Extract conditions from the rule body
        # "age(A,g4),bmi(A,g0),insulin(A,g1)"  # noqa: ERA001
        body = rule.split(":-")[1]
        conditions = re.findall(r"(\w+)\(A,(g\d)\)", body)

        match = True
        for feature, bin_id in conditions:
            if str(patient_data.get(feature)) != bin_id:
                match = False
                break

        if match:
            boundary_conditions.append(body)
    return boundary_conditions

# -----------------------------------------------------------------------------


st.set_page_config(page_title="Medical xAI", layout="wide")
st.title("Medical Diagnosis with PyGol Explainability")

metadata, rules = load_resources()

uploaded_file = st.file_uploader("Upload Symbolic Test Data (CSV)", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    col1, col2 = st.columns([1, 2])

    with col1:
        patient_idx = st.selectbox("Select Patient Index", options=df.index)
        patient_data = df.loc[patient_idx].to_dict()
        st.write("**Patient Summary:**")
        st.json(patient_data)

    with col2:
        active_rules = get_rules(patient_data, rules["hypothesis"])

        if active_rules:
            st.error("**Positive Diagnosis Predicted**")

            for i, conditions in enumerate(active_rules):
                with st.expander(f"Rule {i+1} satisfied:"):
                    for feature, bin_id in conditions:
                        range_str = metadata.get(feature, {}).get(bin_id, "Unknown range")
                        st.write(f"- **{feature.capitalize()}**: `{range_str}`")
        else:
            st.success("**Negative Diagnosis Predicted**")
            st.write("No rules were satisfied for this patient.")
