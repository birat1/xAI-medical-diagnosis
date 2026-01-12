# Towards Explainable Medical Diagnosis: Counterfactual Explanations for Clinical Decisions

This repository contains the code for my final year dissertation. It focuses on generating counterfactual explanations to prove transparent and interpretable clinical decisions.

## Overview

**Goal**: To develop a system that not only predicts medical conditions but explains them through "what-if" scenarios (counterfactuals).  
**Covers**: Random Forest, Multi-Layer Perception, [SHAP](https://github.com/shap/shap), [LIME](https://github.com/marcotcr/lime), [DiCE](https://github.com/interpretml/DiCE), [PyGol](https://github.com/danyvarghese/PyGol)  
**Data**: Medical datasets (e.g., [Pima Indians Diabetes](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database))

## Installation

This project uses uv for managing packages.

```bash
# Install dependencies
uv sync
```

## Usage

###

#### Preprocessing

```py
uv run preprocessing.py --input [../data/raw/diabetes.csv] --output [../data/processed]
```

#### Model Training

```py
uv run models.py
```

#### Threshold optimisation

```py
uv run optimise_threshold.py
```

#### Model Evaluation

```py
uv run evaluate_models.py
```

#### Interpret

```py
uv run interpretability.py
```

## Results

Results are stored in the [results](https://github.com/birat1/xAI-medical-diagnosis/tree/master/results) folder.