import hydra
from data.dataset import load_dataset
from model.boosting_tree import run_kfold_lightgbm
from omegaconf import DictConfig


@hydra.main(config_path="../../config/train/", config_name="lgbm_train.yml")
def _main(cfg: DictConfig):
    path = hydra.utils.to_absolute_path(cfg.dataset.path) + "/"
    train, test = load_dataset(path)
    # train = pd.read_pickle(cfg["path"] + "train.pkl")
    # test = pd.read_pickle(cfg["path"] + "test.pkl")
    # Split features and target
    X = train.drop(["row_id", "target", "time_id"], axis=1)
    y = train["target"]

    y_pred = test[["row_id"]]
    X_test = test.drop(["time_id", "row_id"], axis=1)

    params = {
        "bagging_fraction": 0.9968055573360797,
        "bagging_freq": 86,
        "bagging_seed": 42,
        "boosting": "gbdt",
        "drop_seed": 42,
        "feature_fraction": 0.20775067071381137,
        "feature_fraction_bynode": 0.8316171499590596,
        "feature_fraction_seed": 42,
        "lambda_l1": 6.537566377843428,
        "lambda_l2": 5.810256006823,
        "learning_rate": 0.11927372203947056,
        "max_depth": 7,
        "min_data_in_leaf": 912,
        "min_sum_hessian_in_leaf": 46.44782425002911,
        "n_jobs": -1,
        "num_leaves": 938,
        "objective": "rmse",
        "seed": 42,
        "verbosity": -1,
    }

    lgb_oof, lgb_preds = run_kfold_lightgbm(
        cfg.model.fold, X, y, X_test, params, cfg.model.verbose
    )

    y_pred = y_pred.assign(target=lgb_preds)
    y_pred.to_csv("submission.csv", index=False)


if __name__ == "__main__":
    _main()
