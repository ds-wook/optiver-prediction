from functools import partial

import hydra
import pandas as pd
from omegaconf import DictConfig
from tune.bayesian import BayesianOptimizer, group_lgbm_objective, lgbm_objective


@hydra.main(config_path="../../config/optimization/", config_name="optiver-optim.yaml")
def _main(cfg: DictConfig):
    path = hydra.utils.to_absolute_path(cfg.dataset.path) + "/"
    train = pd.read_pickle(path + cfg.dataset.train)

    # Split features and target
    X = train.drop(["row_id", "target", "time_id"], axis=1)
    y = train["target"]

    objective = (
        partial(lgbm_objective, X=X, y=y, n_fold=cfg.model.fold)
        if cfg.model.fold_name == "kf"
        else partial(
            group_lgbm_objective,
            X=X,
            y=y,
            groups=train["time_id"],
            n_fold=cfg.model.fold,
        )
    )

    bayesian_optim = BayesianOptimizer(objective)
    study = bayesian_optim.build_study(trials=cfg.optimization.trials)
    bayesian_optim.lgbm_save_params(study, cfg.optimization.params)


if __name__ == "__main__":
    _main()
