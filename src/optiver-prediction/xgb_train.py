import hydra
import pandas as pd
from hydra.utils import to_absolute_path
from model.boosting_tree import run_group_kfold_xgboost, run_kfold_xgboost
from omegaconf import DictConfig
from utils.utils import rmspe


@hydra.main(config_path="../../config/train/", config_name="lgbm_train.yml")
def _main(cfg: DictConfig):
    path = to_absolute_path(cfg.dataset.path) + "/"
    train = pd.read_pickle(path + "fea_train_best.pkl")
    # Split features and target
    X = train.drop(["row_id", "target", "time_id"], axis=1)
    y = train["target"]
    # Transform stock id to a numeric value
    X["stock_id"] = X["stock_id"].astype(int)

    # Hyperparammeters (optimized)
    params = {
        "max_depth": 6,
        "n_estimators": 1030,
        "eta": 0.14411433835252993,
        "subsample": 0.7,
        "colsample_bytree": 0.6,
        "min_child_weight": 72,
        "reg_lambda": 0.0011282775827873572,
        "reg_alpha": 0.034971544950363434,
    }
    xgb_oof = (
        run_group_kfold_xgboost(
            cfg.model.fold, X, y, train["time_id"], params, cfg.model.verbose
        )
        if cfg.model.fold_name == "group"
        else run_kfold_xgboost(cfg.model.fold, X, y, params, cfg.model.verbose)
    )

    print(f"RMSPE's Score: {rmspe(y, xgb_oof)}")


if __name__ == "__main__":
    _main()
