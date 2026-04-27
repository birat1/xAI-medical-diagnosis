"""Tune hyperparameters using training data."""
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import pandas as pd
from scipy.stats import randint, uniform
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.tree import DecisionTreeClassifier

from models import load_data, predict_probs, set_seed, train_mlp

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

MODELS_DIR = Path("../models")
METRICS_DIR = Path("../results/metrics")
METRICS_DIR.mkdir(parents=True, exist_ok=True)

def tune_dt(x_train: pd.DataFrame, y_train: np.ndarray) -> dict[str, Any]:
    """Tune hyperparameters for Decision Tree."""
    dt = DecisionTreeClassifier(
        class_weight="balanced",
        random_state=42,
    )

    param_dist = {
        "max_depth": [3, 5, 7, 10, 15],
        "min_samples_split": randint(2, 30),
        "min_samples_leaf": randint(1, 15),
        "criterion": ["gini", "entropy", "log_loss"],
        "ccp_alpha": uniform(0.0, 0.02),
    }

    search = RandomizedSearchCV(
        estimator=dt,
        param_distributions=param_dist,
        n_iter=50,
        scoring="average_precision",
        cv=5,
        n_jobs=-1,
        verbose=2,
        random_state=42,
    )

    search.fit(x_train, y_train)

    logger.info(f"Best DT params: {search.best_params_}")
    logger.info(f"Best DT CV AUPRC: {search.best_score_:.4f}")

    return search.best_params_

def tune_rf(x_train: pd.DataFrame, y_train: np.ndarray) -> dict[str, Any]:
    """Tune hyperparameters for Random Forest."""
    rf = RandomForestClassifier(
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    )

    param_dist = {
        "n_estimators": randint(100, 600),
        "max_depth": [5, 8, 10, 15, 20],
        "min_samples_split": randint(2, 25),
        "min_samples_leaf": randint(1, 15),
        "max_features": ["sqrt", "log2", None],
        "bootstrap": [True, False],
    }

    search = RandomizedSearchCV(
        estimator=rf,
        param_distributions=param_dist,
        n_iter=75,
        scoring="average_precision",
        cv=5,
        n_jobs=-1,
        verbose=2,
        random_state=42,
    )

    search.fit(x_train, y_train)

    logger.info(f"Best RF params: {search.best_params_}")
    logger.info(f"Best RF CV AUPRC: {search.best_score_:.4f}")

    return search.best_params_

def tune_mlp(x_train: pd.DataFrame, y_train: np.ndarray, n_trials: int = 50) -> dict[str, Any]:
    """Tune hyperparameters for MLP."""
    x_mlp_train, x_mlp_cv, y_mlp_train, y_mlp_cv = train_test_split(
        x_train, y_train,
        test_size=0.2,
        stratify=y_train,
        random_state=42,
    )

    def objective(trial: optuna.Trial) -> float:
        n_layers = trial.suggest_int("n_layers", 1, 3)

        hidden_layers = []
        for i in range(n_layers):
            hidden_layers.append(  # noqa: PERF401
                trial.suggest_categorical(f"hidden_size_{i}", [16, 32, 64, 128]),
            )

        params = {
            "lr": trial.suggest_float("lr", 1e-4, 1e-3, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True),
            "dropout": trial.suggest_float("dropout", 0.1, 0.4),
            "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64]),
            "hidden_layers": hidden_layers,
        }

        temp_path = MODELS_DIR / f"temp_mlp_{trial.number}.pth"

        model = train_mlp(
            x_train=x_mlp_train,
            y_train=y_mlp_train,
            x_val=x_mlp_cv,
            y_val=y_mlp_cv,
            epochs=150,
            save_path=temp_path,
            trial=trial,
            **params,
        )

        probs = predict_probs(model, x_mlp_cv, is_pytorch=True)
        auprc = average_precision_score(y_mlp_cv, probs)

        temp_path.unlink(missing_ok=True)

        return auprc

    study = optuna.create_study(
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=10,
            interval_steps=5,
        ),
    )
    study.optimize(objective, n_trials=n_trials)

    return study.best_params

if __name__ == "__main__":
    set_seed(42)

    x_train, _, _, y_train, _, _ = load_data()

    best_params = {
        "decision_tree": tune_dt(x_train, y_train),
        "random_forest": tune_rf(x_train, y_train),
        "mlp": tune_mlp(x_train, y_train, n_trials=100),
    }

    with (METRICS_DIR / "hyperparameters.json").open("w") as f:
        json.dump(best_params, f, indent=2)
    logger.info(f"Best hyperparameters saved to {METRICS_DIR / 'hyperparameters.json'}")
