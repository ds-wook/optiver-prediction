import hydra
import pandas as pd
from hydra.utils import to_absolute_path
from model.boosting_tree import run_group_kfold_lightgbm, run_kfold_lightgbm
from omegaconf import DictConfig


@hydra.main(config_path="../../config/train/", config_name="lgbm_train.yml")
def _main(cfg: DictConfig):
    path = to_absolute_path(cfg.dataset.path) + "/"
    train = pd.read_pickle(path + "fea_train_best.pkl")
    test = pd.read_pickle(path + "fea_test_best.pkl")
    # Split features and target
    X = train.drop(["row_id", "target", "time_id"], axis=1)
    y = train["target"]
    X_test = test.drop(["row_id", "time_id"], axis=1)
    # Transform stock id to a numeric value
    X["stock_id"] = X["stock_id"].astype(int)
    X_test["stock_id"] = X_test["stock_id"].astype(int)
    # Hyperparammeters (optimized)
    seed = 2021
    params = {
        "learning_rate": 0.1,
        "lambda_l1": 2,
        "lambda_l2": 7,
        "num_leaves": 800,
        "min_sum_hessian_in_leaf": 20,
        "feature_fraction": 0.8,
        "feature_fraction_bynode": 0.8,
        "bagging_fraction": 0.9,
        "bagging_freq": 42,
        "min_data_in_leaf": 700,
        "max_depth": 4,
        "seed": seed,
        "feature_fraction_seed": seed,
        "bagging_seed": seed,
        "drop_seed": seed,
        "data_random_seed": seed,
        "objective": "rmse",
        "boosting": "gbdt",
        "verbosity": -1,
        "n_jobs": -1,
    }
    lgb_oof, lgb_preds = (
        run_group_kfold_lightgbm(
            cfg.model.fold, X, y, X_test, train["time_id"], params, cfg.model.verbose
        )
        if cfg.model.fold_name == "group"
        else run_kfold_lightgbm(cfg.model.fold, X, y, X_test, params, cfg.model.verbose)
    )

    # Save test predictions
    test["target"] = lgb_preds
    test[["row_id", "target"]].to_csv("submission.csv", index=False)


if __name__ == "__main__":
    _main()
