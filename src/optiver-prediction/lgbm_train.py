import argparse

from data.dataset import load_dataset
from model.boosting_tree import run_kfold_lightgbm


def define_argparser():
    parse = argparse.ArgumentParser("Train!")
    parse.add_argument(
        "--path", type=str, default="../../input/optiver-realized-volatility-prediction"
    )
    parse.add_argument("--verbose", type=int, default=False)
    parse.add_argument("--token", type=str, default="None")
    parse.add_argument("--fold", type=int, default=5)
    args = parse.parse_args()
    return args


def _main(args: argparse.Namespace):
    df_train, df_test = load_dataset(args.path)
    X = df_train.drop(["row_id", "target"], axis=1)
    y = df_train["target"]

    y_pred = df_test[["row_id"]]
    X_test = df_test.drop(["time_id", "row_id"], axis=1)

    # training lightgbm
    params = {
        "objective": "rmse",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "early_stopping_rounds": 30,
        "learning_rate": 0.01,
        "lambda_l1": 1,
        "lambda_l2": 1,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
    }

    lgb_oof, lgb_preds = run_kfold_lightgbm(
        args.fold, X, y, X_test, params, args.token, args.verbose
    )

    y_pred = y_pred.assign(target=lgb_preds)
    y_pred.to_csv("submission.csv", index=False)


if __name__ == "__main__":
    args = define_argparser()
    _main(args)
