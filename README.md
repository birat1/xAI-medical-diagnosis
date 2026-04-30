# Towards Explainable Medical Diagnosis: Counterfactual Explanations for Clinical Decisions

This repository contains the code for my final year dissertation.

The project develops and evaluates ML models for medical diagnosis, with a focus on interpretability and explainability through feature attribution methods and counterfactual explanations.

## Overview

**Goal**: To develop a pipeline that combines predictive performance with interpretability. The system contains feature attribution methods (SHAP, LIME) and counterfactual explanations (DiCE, PyGol).  
**Covers**: Random Forest, Decision Trees, Multi-Layer Perception (MLP), [SHAP](https://github.com/shap/shap), [LIME](https://github.com/marcotcr/lime), [DiCE](https://github.com/interpretml/DiCE), [PyGol](https://github.com/danyvarghese/PyGol)  
**Data**: Medical datasets (e.g., [Diabetes](https://www.kaggle.com/datasets/johndasilva/diabetes))

## Installation

This project uses [uv](https://docs.astral.sh/uv/) for managing packages.

```bash
# Install dependencies
uv sync
```

## Usage

Run the following steps in order:

### Preprocessing

```bash
uv run preprocessing.py --input ../data/raw/diabetes.csv --output ../data/processed
```

#### Hyperparameter Tuning

```bash
uv run tune_hyperparameters.py
```

#### Model Training

```bash
uv run train_models.py
```

#### Threshold optimisation

```bash
uv run tune_thresholds.py
```

#### Model Evaluation

```bash
uv run evaluate_models.py
```

#### Interpretability (SHAP, LIME)

```bash
uv run explain_models.py
```

### Explainability (DiCE, PyGol)

```bash
uv run generate_counterfactuals.py
```

### Interpreting Counterfactuals

```bash
uv run interpret_counterfactuals.py
```

### Streamlit Interface

```bash
uv run streamlit run app.py
```

## Results

Detailed metrics, plots, and explanations are stored in the [results](https://github.com/birat1/xAI-medical-diagnosis/tree/master/results) folder.
