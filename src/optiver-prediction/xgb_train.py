import hydra
import pandas as pd
from data.dataset import add_tau_feature, create_agg_features
from hydra.utils import to_absolute_path
from model.boosting_tree import run_group_kfold_xgboost, run_kfold_xgboost
from omegaconf import DictConfig
from utils.utils import rmspe


@hydra.main(config_path="../../config/train/", config_name="xgb.yaml")
def _main(cfg: DictConfig):
    path = to_absolute_path(cfg.dataset.path) + "/"
    train = pd.read_pickle(path + cfg.dataset.train)
    test = pd.read_pickle(path + cfg.dataset.test)
    train, test = add_tau_feature(train, test)
    train, test = create_agg_features(train, test, path)
    print(train.shape, test.shape)

    # Split features and target
    X = train.drop(["row_id", "target", "time_id"], axis=1)
    y = train["target"]
    # Transform stock id to a numeric value
    X["stock_id"] = X["stock_id"].astype(int)

    # Hyperparammeters (optimized)
    params = dict(cfg.params)
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
