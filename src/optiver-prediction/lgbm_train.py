import hydra
import pandas as pd
from hydra.utils import to_absolute_path
from model.boosting_tree import run_kfold_lightgbm
from omegaconf import DictConfig, OmegaConf


@hydra.main(config_path="../../config/train/", config_name="lgbm_train.yml")
def _main(cfg: DictConfig):
    path = to_absolute_path(cfg.dataset.path) + "/"
    train = pd.read_pickle(path + "train.pkl")
    test = pd.read_pickle(path + "test.pkl")
    # Split features and target
    X = train.drop(["row_id", "target", "time_id"], axis=1)
    y = train["target"]

    y_pred = test[["row_id"]]
    X_test = test.drop(["time_id", "row_id"], axis=1)
    params_path = to_absolute_path("../../parameters/best_lgbm_param.yml")
    params = OmegaConf.load(params_path)

    lgb_oof, lgb_preds = run_kfold_lightgbm(
        cfg.model.fold, X, y, X_test, params, cfg.model.verbose
    )

    y_pred = y_pred.assign(target=lgb_preds)
    y_pred.to_csv("submission.csv", index=False)


if __name__ == "__main__":
    _main()
