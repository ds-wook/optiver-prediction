import hydra
import pandas as pd
from hydra.utils import to_absolute_path
from model.boosting_tree import run_group_kfold_lightgbm, run_kfold_lightgbm
from omegaconf import DictConfig


@hydra.main(config_path="../../config/train/", config_name="train.yaml")
def _main(cfg: DictConfig):
    path = to_absolute_path(cfg.dataset.path) + "/"
    train = pd.read_pickle(path + cfg.dataset.train)
    test = pd.read_pickle(path + cfg.dataset.test)

    # Split features and target
    X = train.drop(["row_id", "target", "time_id"], axis=1)
    y = train["target"]
    X_test = test.drop(["row_id", "time_id"], axis=1)

    # Transform stock id to a numeric value
    X["stock_id"] = X["stock_id"].astype(int)
    X_test["stock_id"] = X_test["stock_id"].astype(int)

    # Hyperparammeters (optimized)
    params = dict(cfg.params)

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
