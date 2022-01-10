import hydra
import neptune.new as neptune
import pandas as pd
from hydra.utils import to_absolute_path
from omegaconf import DictConfig

from data.dataset import add_tau_feature, create_agg_features
from model.gbdt import LightGBMTrainer
from utils.utils import rmspe, timer


@hydra.main(config_path="../config/train/", config_name="lgbm.yaml")
def _main(cfg: DictConfig):
    path = to_absolute_path(cfg.dataset.path) + "/"
    train = pd.read_pickle(path + cfg.dataset.train)
    test = pd.read_pickle(path + cfg.dataset.test)
    model_name = cfg.model.select

    if model_name == "lightgbm":
        # make experiment tracking
        run = neptune.init(
            project=cfg.experiment.project, tags=list(cfg.experiment.tags.lightgbm)
        )

        train, test = add_tau_feature(train, test)
        train, test = create_agg_features(train, test, path)
        train["log_return1_realized_volatility_is_high"] = train[
            "log_return1_realized_volatility"
        ].apply(lambda x: 0 if 0.0001 <= x <= 0.0003 else 1)

        test["log_return1_realized_volatility_is_high"] = test[
            "log_return1_realized_volatility"
        ].apply(lambda x: 0 if 0.0001 <= x <= 0.0003 else 1)

        # Split features and target
        X = train.drop(["row_id", "target", "time_id"], axis=1)
        y = train["target"]
        X_test = test.drop(["row_id", "time_id"], axis=1)
        groups = train["time_id"]

        # Transform stock id to a numeric value
        X["stock_id"] = X["stock_id"].astype(int)
        X_test["stock_id"] = X_test["stock_id"].astype(int)

        with timer("LightGBM Learning"):
            # model train
            lgbm_trainer = LightGBMTrainer(
                params=cfg.model.lightgbm.params,
                run=run,
                fold=cfg.model.fold,
                metric=rmspe,
            )
            lgbm_trainer.train(X, y, groups)
            lgbm_preds = lgbm_trainer.predict(X_test)

        # Save test predictions
        test["target"] = lgbm_preds
        test[["row_id", "target"]].to_csv("submission.csv", index=False)


if __name__ == "__main__":
    _main()
