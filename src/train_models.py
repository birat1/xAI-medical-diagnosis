"""Train models using tuned hyperparameters."""
import logging

from sklearn.model_selection import train_test_split

from models import load_data, load_hyperparameters, set_seed, train_dt, train_mlp, train_rf

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    set_seed(42)
    logger.info("-" * 50)

    x_train, _, _, y_train, _, _ = load_data()
    best_params = load_hyperparameters()

    logger.info("Training DT with best hyperparameters...")
    train_dt(x_train=x_train, y_train=y_train, params=best_params["decision_tree"])

    logger.info("Training RF with best hyperparameters...")
    train_rf(x_train=x_train, y_train=y_train, params=best_params["random_forest"])

    logger.info("Training MLP with best hyperparameters...")
    x_mlp_train, x_mlp_stop, y_mlp_train, y_mlp_stop = train_test_split(
        x_train, y_train,
        test_size=0.2,
        stratify=y_train,
        random_state=42,
    )

    mlp_params = best_params["mlp"]
    hidden_layers = [
        mlp_params[f"hidden_size_{i}"]
        for i in range(mlp_params["n_layers"])
    ]

    train_mlp(
        x_train=x_mlp_train,
        y_train=y_mlp_train,
        x_val=x_mlp_stop,
        y_val=y_mlp_stop,
        epochs=500,
        hidden_layers=hidden_layers,
        dropout=mlp_params["dropout"],
        lr=mlp_params["lr"],
        weight_decay=mlp_params["weight_decay"],
        batch_size=mlp_params["batch_size"],
    )

    logger.info("All models trained and saved.")
