from functools import partial

import hydra
from data.dataset import load_dataset
from omegaconf import DictConfig
from optimization.bayesian import BayesianOptimizer, lgbm_objective


@hydra.main(config_path="../../config/optimization/", config_name="optiver-optim.yml")
def _main(cfg: DictConfig):
    path = hydra.utils.to_absolute_path(cfg.dataset.path) + "/"
    train, test = load_dataset(path)
    # train = pd.read_pickle(cfg["path"] + "train.pkl")
    # Split features and target
    X = train.drop(["row_id", "target", "time_id"], axis=1)
    y = train["target"]

    objective = partial(lgbm_objective, X=X, y=y, n_fold=cfg.model.fold)
    bayesian_optim = BayesianOptimizer(objective)
    study = bayesian_optim.build_study(trials=cfg.optimization.trials)
    bayesian_optim.lgbm_save_params(study, cfg.optimization.params)


if __name__ == "__main__":
    _main()
