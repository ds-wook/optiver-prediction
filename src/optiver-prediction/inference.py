import hydra
import pandas as pd
from hydra.utils import to_absolute_path
from model.boosting_tree import load_lightgbm_model
from omegaconf import DictConfig


@hydra.main(config_path="../../config/train/", config_name="train.yaml")
def _main(cfg: DictConfig):
    path = to_absolute_path(cfg.dataset.path) + "/"
    train = pd.read_pickle(path + cfg.dataset.train)
    test = pd.read_pickle(path + cfg.dataset.test)
    print(train.shape, test.shape)

    # Split features and target
    X = train.drop(["row_id", "target", "time_id"], axis=1)
    y = train["target"]
    X_test = test.drop(["row_id", "time_id"], axis=1)
    groups = train["time_id"]

    # Transform stock id to a numeric value
    X["stock_id"] = X["stock_id"].astype(int)
    X_test["stock_id"] = X_test["stock_id"].astype(int)

    lgb_preds = load_lightgbm_model(5, X, y, X_test, groups=groups)
    # Save test predictions
    test["target"] = lgb_preds
    test[["row_id", "target"]].to_csv("submission.csv", index=False)


if __name__ == "__main__":
    _main()
