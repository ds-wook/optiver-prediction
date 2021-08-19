import hydra
import pandas as pd
from hydra.utils import to_absolute_path
from model.regression import run_kfold_rgf
from omegaconf import DictConfig
from utils.utils import rmspe


@hydra.main(config_path="../../config/train/", config_name="train.yaml")
def _main(cfg: DictConfig):
    path = to_absolute_path(cfg.dataset.path) + "/"
    train = pd.read_csv(path + "train_with_features.csv")
    # Split features and target
    X = train.drop(["row_id", "target", "time_id"], axis=1)
    y = train["target"]
    # Transform stock id to a numeric value
    X["stock_id"] = X["stock_id"].astype(int)

    # Hyperparammeters (optimized)
    params = {
        "max_leaf": 400,
        "algorithm": "RGF_Sib",
        "test_interval": 100,
        "verbose": True,
    }
    rgf_oof = run_kfold_rgf(cfg.model.fold, X, y, params)

    print(f"RMSPE's Score: {rmspe(y, rgf_oof)}")


if __name__ == "__main__":
    _main()
