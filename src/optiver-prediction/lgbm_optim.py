import argparse
from functools import partial

import pandas as pd
from optimization.bayesian import BayesianOptimizer, lgbm_objective


def define_argparser():
    parse = argparse.ArgumentParser("Optimize")
    parse.add_argument("--fold", type=int, default=10)
    parse.add_argument("--trials", type=int, default=100)
    parse.add_argument("--params", type=str, default="params.json")
    args = parse.parse_args()
    return args


def _main(args: argparse.Namespace):
    train = pd.read_pickle("../../input/train.pkl")
    # Split features and target
    X = train.drop(["row_id", "target", "time_id"], axis=1)
    y = train["target"]

    objective = partial(lgbm_objective, X=X, y=y, n_fold=args.fold)
    bayesian_optim = BayesianOptimizer(objective)
    study = bayesian_optim.build_study(trials=args.trials)
    bayesian_optim.lgbm_save_params(study, args.params)


if __name__ == "__main__":
    args = define_argparser()
    _main(args)
