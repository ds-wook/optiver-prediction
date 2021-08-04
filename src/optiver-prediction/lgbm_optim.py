import argparse
from functools import partial
from typing import Any, Dict

import pandas as pd
import yaml
from optimization.bayesian import BayesianOptimizer, lgbm_objective


def define_argparser():
    parse = argparse.ArgumentParser("Optimize")
    parse.add_argument("--fold", type=int, default=10)
    parse.add_argument("--trials", type=int, default=100)
    parse.add_argument("--params", type=str, default="params.json")
    args = parse.parse_args()
    return args


def _main(cfg: Dict[str, Any]):
    train = pd.read_pickle(cfg["path"] + "train.pkl")
    # Split features and target
    X = train.drop(["row_id", "target", "time_id"], axis=1)
    y = train["target"]

    objective = partial(lgbm_objective, X=X, y=y, n_fold=cfg["fold"])
    bayesian_optim = BayesianOptimizer(objective)
    study = bayesian_optim.build_study(trials=cfg["trial"])
    bayesian_optim.lgbm_save_params(study, cfg["params"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        help="configuration file *.yml",
        type=str,
        required=False,
        default="../../config/optimization/optiver-optim.yml",
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.load(f, yaml.FullLoader)
    _main(cfg)
