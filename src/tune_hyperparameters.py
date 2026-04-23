"""Hyperparameter tuning using 5-fold CV."""
import json
import logging
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.metrics import average_precision_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

from models import MLP

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path("../data/processed")
RESULTS_DIR = Path("../results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed: int = 42) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.default_rng(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_data() -> tuple[pd.DataFrame, np.ndarray]:
    """Load development data for 5-fold CV."""
    x_dev = pd.read_csv(DATA_DIR / "x_dev.csv")
    y_dev = pd.read_csv(DATA_DIR / "y_dev.csv").to_numpy().ravel()

    return x_dev, y_dev

def clean_params(params: dict) -> dict:
    """Clean parameter names from RandomizedSearchCV."""
    return {k.replace("model__", ""): v for k, v in params.items()}

def fit_preprocessor(x_train_fold: pd.DataFrame, seed: int = 42) -> tuple[IterativeImputer, StandardScaler, np.ndarray]:
    """Fit imputer and scaler on training fold."""
    imputer = IterativeImputer(random_state=seed)
    scaler = StandardScaler()

    x_train_imputed = imputer.fit_transform(x_train_fold)
    x_train_scaled = scaler.fit_transform(x_train_imputed)

    return imputer, scaler, x_train_scaled

def transform_fold(x_fold: pd.DataFrame, imputer: IterativeImputer, scaler: StandardScaler) -> np.ndarray:
    """Transform fold using fitted preprocessor."""
    x_imputed = imputer.transform(x_fold)
    x_scaled = scaler.transform(x_imputed)

    return x_scaled

###
###
###

def search_dt(x_dev: pd.DataFrame, y_dev: np.ndarray, seed: int = 42) -> dict[str, object]:
    """Search DT hyperparameters."""
    logger.info("Starting Decision Tree hyperparameter search...")

    pipeline = Pipeline([
        ("imputer", IterativeImputer(random_state=seed)),
        ("scaler", StandardScaler()),
        ("model", DecisionTreeClassifier(class_weight="balanced", random_state=seed)),
    ])

    params = {
        "model__max_depth": [3, 5, 7, 10],
        "model__min_samples_split": [2, 5, 8, 10, 15],
        "model__min_samples_leaf": [1, 2, 5, 8],
        "model__criterion": ["gini", "entropy"],
        "model__ccp_alpha": [0.0, 0.0005, 0.001, 0.005],
    }

    search = RandomizedSearchCV(
        pipeline,
        param_distributions=params,
        n_iter=25,
        scoring="average_precision",
        cv=5,
        random_state=seed,
        n_jobs=-1,
        verbose=1,
    )

    search.fit(x_dev, y_dev)

    logger.info(f"Best DT AUPRC: {search.best_score_:.4f}")
    logger.info(f"Best DT params: {clean_params(search.best_params_)}")

    return {
        "best": {
            "params": clean_params(search.best_params_),
            "mean_auprc": search.best_score_,
        },
        "cv_results": {
            "mean_test_score": search.cv_results_["mean_test_score"].tolist(),
            "params": [
                clean_params(params) for params in search.cv_results_["params"]
            ],
        },
    }

def search_rf(x_dev: pd.DataFrame, y_dev: np.ndarray, seed: int = 42) -> dict[str, object]:
    """Search RF hyperparameters."""
    logger.info("Starting Random Forest hyperparameter search...")

    pipeline = Pipeline([
        ("imputer", IterativeImputer(random_state=seed)),
        ("scaler", StandardScaler()),
        ("model", RandomForestClassifier(random_state=seed, n_jobs=-1)),
    ])

    params = {
        "model__n_estimators": [100, 200, 300, 500],
        "model__max_depth": [5, 8, 10, 12, 15],
        "model__min_samples_split": [2, 5, 8, 10, 15],
        "model__min_samples_leaf": [1, 2, 3, 5, 8],
        "model__max_features": ["sqrt", "log2"],
        "model__class_weight": ["balanced", "balanced_subsample"],
    }

    search = RandomizedSearchCV(
        pipeline,
        param_distributions=params,
        n_iter=25,
        scoring="average_precision",
        cv=5,
        random_state=seed,
        n_jobs=-1,
        verbose=1,
        refit=True,
    )

    search.fit(x_dev, y_dev)

    logger.info(f"Best RF AUPRC: {search.best_score_:.4f}")
    logger.info(f"Best RF params: {clean_params(search.best_params_)}")

    return {
        "best": {
            "params": clean_params(search.best_params_),
            "mean_auprc": search.best_score_,
        },
        "cv_results": {
            "mean_test_score": search.cv_results_["mean_test_score"].tolist(),
            "params": [
                clean_params(params) for params in search.cv_results_["params"]
            ],
        },
    }

def train_mlp_once(  # noqa: PLR0913
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray,
        y_val: np.ndarray,
        params: dict,
        seed: int = 42,
    ) -> float:
    """Train one MLP fold and return validation AUPRC."""
    set_seed(seed)

    model = MLP(x_train.shape[1]).to(device)

    # 1. Weighted Loss: handle class imbalance
    pos = float((y_train == 1).sum())
    neg = float((y_train == 0).sum())

    if pos == 0:
        raise ValueError("No positive samples in training fold.")  # noqa: EM101, TRY003

    pos_weight = torch.tensor([neg / pos], dtype=torch.float32, device=device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimiser = optim.Adam(model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"])

    # 2. LR Scheduler: reduces learning rate on plateau of validation loss
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimiser, mode="min", factor=0.5, patience=6, min_lr=1e-6)

    # Convert data to tensors
    x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)

    x_val_tensor = torch.tensor(x_val, dtype=torch.float32, device=device)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32, device=device).view(-1, 1)

    train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=params["batch_size"], shuffle=True, drop_last=False)

    best_state = None
    best_val_auprc = -float("inf")
    early_stop_patience = params.get("patience", 20)
    patience_counter = 0

    for epoch in range(1, params["epochs"] + 1):
        # Training phase
        model.train()
        train_loss = 0.0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimiser.zero_grad(set_to_none=True)
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimiser.step()
            train_loss += loss.item() * batch_x.size(0)

        train_loss /= len(train_loader.dataset)

        # 3. Validation phase for early stopping and LR scheduling
        model.eval()
        with torch.no_grad():
            val_logits = model(x_val_tensor)
            val_loss = criterion(val_logits, y_val_tensor)

            val_probs = torch.sigmoid(val_logits).cpu().numpy().ravel()
            val_auprc = average_precision_score(y_val, val_probs)

        # Step the scheduler based on validation loss
        scheduler.step(val_loss)

        # Save the best model and stop if no improvement
        if val_auprc > best_val_auprc:
            best_val_auprc = val_auprc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                break

    return float(best_val_auprc)

def evaluate_mlp_params(
        x_dev: pd.DataFrame,
        y_dev: np.ndarray,
        params: dict,
        n_splits: int = 5,
        seed: int = 42,
    ) -> dict[str, object]:
    """Evaluate MLP hyperparameters using 5-fold CV."""
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_auprc: list[float] = []

    for fold, (train_idx, val_idx) in enumerate(cv.split(x_dev, y_dev), start=1):
        logger.info(f"[MLP SEARCH] Fold {fold}/{n_splits}")

        x_train_fold, y_train_fold = x_dev.iloc[train_idx].copy(), y_dev[train_idx]
        x_val_fold, y_val_fold = x_dev.iloc[val_idx].copy(), y_dev[val_idx]

        imputer, scaler, x_train_fold_scaled = fit_preprocessor(x_train_fold, seed)
        x_val_fold_scaled = transform_fold(x_val_fold, imputer, scaler)

        fold_score = train_mlp_once(
            x_train=x_train_fold_scaled,
            y_train=y_train_fold,
            x_val=x_val_fold_scaled,
            y_val=y_val_fold,
            params=params,
            seed=seed,
        )
        fold_auprc.append(fold_score)

    return {
        "params": params,
        "fold_auprc": fold_auprc,
        "mean_auprc": np.mean(fold_auprc),
    }

def search_mlp(x_dev: pd.DataFrame, y_dev: np.ndarray, seed: int = 42) -> dict[str, object]:
    """Manual search for MLP hyperparameters."""
    logger.info("Starting MLP hyperparameter search...")

    candidates = [
        {"lr": 1e-3, "weight_decay": 1e-3, "batch_size": 64, "epochs": 300, "patience": 20},
        {"lr": 5e-4, "weight_decay": 1e-3, "batch_size": 64, "epochs": 300, "patience": 20},
        {"lr": 1e-4, "weight_decay": 1e-3, "batch_size": 64, "epochs": 300, "patience": 20},
        {"lr": 1e-3, "weight_decay": 1e-4, "batch_size": 64, "epochs": 300, "patience": 20},
        {"lr": 5e-4, "weight_decay": 1e-4, "batch_size": 64, "epochs": 300, "patience": 20},
        {"lr": 1e-3, "weight_decay": 0.0, "batch_size": 64, "epochs": 300, "patience": 20},
        {"lr": 1e-3, "weight_decay": 1e-3, "batch_size": 32, "epochs": 300, "patience": 20},
        {"lr": 5e-4, "weight_decay": 1e-3, "batch_size": 32, "epochs": 300, "patience": 20},
    ]

    best_result = None
    all_results = []

    for i, params in enumerate(candidates, start=1):
        logger.info(f"[MLP SEARCH] Candidate {i}/{len(candidates)} | {params}")

        result = evaluate_mlp_params(x_dev, y_dev, params, n_splits=5, seed=seed)
        all_results.append(result)

        logger.info(f"[MLP SEARCH] Candidate {i} mean AUPRC: {result['mean_auprc']:.4f}")

        if best_result is None or result["mean_auprc"] > best_result["mean_auprc"]:
            best_result = result

    logger.info(f"Best MLP AUPRC: {best_result['mean_auprc']:.4f}")
    logger.info(f"Best MLP params: {best_result['params']}")

    return {
        "best": best_result,
        "all_results": all_results,
    }

def save_params(results: dict[str, object]) -> None:
    """Save hyperparameters to JSON file."""
    output = {
        "dt": {
            "best_params": results["dt"]["best"]["params"],
            "mean_auprc": results["dt"]["best"]["mean_auprc"],
        },
        "rf": {
            "best_params": results["rf"]["best"]["params"],
            "mean_auprc": results["rf"]["best"]["mean_auprc"],
        },
        "mlp": {
            "best_params": results["mlp"]["best"]["params"],
            "mean_auprc": results["mlp"]["best"]["mean_auprc"],
        },
    }

    with (RESULTS_DIR / "tuned_hyperparameters.json").open("w") as f:
        json.dump(output, f, indent=2)

    with (RESULTS_DIR / "tuning_results.json").open("w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Saved tuned hyperparameters to {RESULTS_DIR / 'tuned_hyperparameters.json'}")
    logger.info(f"Saved full tuning results to {RESULTS_DIR / 'tuning_results.json'}")

if __name__ == "__main__":
    """Run hyperparameter tuning for all models."""
    set_seed(42)

    x_dev, y_dev = load_data()

    dt_results = search_dt(x_dev, y_dev, seed=42)
    rf_results = search_rf(x_dev, y_dev, seed=42)
    mlp_results = search_mlp(x_dev, y_dev, seed=42)

    results = {
        "dt": dt_results,
        "rf": rf_results,
        "mlp": mlp_results,
    }

    save_params(results)

    logger.info("\n=== Hyperparameter Tuning Summary ===")
    logger.info(
        f"DT: {dt_results['best']['params']} | "
        f"mean AUPRC: {dt_results['best']['mean_auprc']:.4f}",
    )
    logger.info(
        f"RF: {rf_results['best']['params']} | "
        f"mean AUPRC: {rf_results['best']['mean_auprc']:.4f}",
    )
    logger.info(
        f"MLP: {mlp_results['best']['params']} | "
        f"mean AUPRC: {mlp_results['best']['mean_auprc']:.4f}",
    )
