import warnings
from typing import Callable, Sequence, Union

import lightgbm as lgbm
import neptune.new as neptune
import neptune.new.integrations.optuna as optuna_utils
import numpy as np
import optuna
import pandas as pd
import yaml
from hydra.utils import to_absolute_path
from neptune.new.exceptions import NeptuneMissingApiTokenException
from optuna.integration import LightGBMPruningCallback
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from optuna.study import Study
from optuna.trial import FrozenTrial, Trial
from sklearn.model_selection import GroupKFold, KFold
from utils.utils import feval_RMSPE, rmspe

warnings.filterwarnings("ignore")


class BayesianOptimizer:
    def __init__(
        self, objective_function: Callable[[Trial], Union[float, Sequence[float]]]
    ):
        self.objective_function = objective_function

    def build_study(self, trials: FrozenTrial, verbose: bool = False):
        try:
            run = neptune.init(
                project="ds-wook/optiver-prediction", tags="optimization"
            )

            neptune_callback = optuna_utils.NeptuneCallback(
                run, plots_update_freq=1, log_plot_slice=False, log_plot_contour=False
            )
            sampler = TPESampler(seed=42)
            study = optuna.create_study(
                study_name="TPE Optimization",
                direction="minimize",
                sampler=sampler,
                pruner=MedianPruner(n_startup_trials=10, n_warmup_steps=5),
            )
            study.optimize(
                self.objective_function, n_trials=trials, callbacks=[neptune_callback]
            )
            run.stop()

        except NeptuneMissingApiTokenException:
            sampler = TPESampler(seed=42)
            study = optuna.create_study(
                study_name="optimization", direction="minimize", sampler=sampler
            )
            study.optimize(self.objective_function, n_trials=trials)
        if verbose:
            self.display_study_statistics(study)

        return study

    @staticmethod
    def display_study_statistics(study: Study):
        print("Best trial:")
        trial = study.best_trial
        print("  Value: ", trial.value)
        print("  Params: ")
        for key, value in trial.params.items():
            print(f"    '{key}': {value},")

    @staticmethod
    def lgbm_save_params(study: Study, params_name: str):
        params = study.best_trial.params
        params["seed"] = 42
        params["feature_fraction_seed"] = 42
        params["bagging_seed"] = 42
        params["drop_seed"] = 42
        params["boosting"] = "gbdt"
        params["objective"] = "rmse"
        params["verbosity"] = -1
        params["n_jobs"] = -1

        with open(to_absolute_path("../../config/train/train.yaml")) as f:
            train_dict = yaml.load(f, Loader=yaml.FullLoader)
        train_dict["params"] = params

        with open(to_absolute_path("../../config/train/" + params_name), "w") as p:
            yaml.dump(train_dict, p)


def lgbm_objective(
    trial: FrozenTrial,
    X: pd.DataFrame,
    y: pd.Series,
    n_fold: int,
) -> float:
    params = {
        "learning_rate": trial.suggest_float("learning_rate", 0.1, 0.2),
        "lambda_l1": trial.suggest_float("lambda_l1", 1, 10),
        "lambda_l2": trial.suggest_float("lambda_l2", 1, 10),
        "num_leaves": trial.suggest_int("num_leaves", 512, 1024),
        "min_sum_hessian_in_leaf": trial.suggest_float(
            "min_sum_hessian_in_leaf", 20, 50
        ),
        "feature_fraction": trial.suggest_uniform("feature_fraction", 0.1, 1),
        "feature_fraction_bynode": trial.suggest_uniform(
            "feature_fraction_bynode", 0.1, 1
        ),
        "bagging_fraction": trial.suggest_uniform("bagging_fraction", 0.1, 1),
        "bagging_freq": trial.suggest_int("bagging_freq", 35, 100),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 512, 1024),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "seed": 42,
        "feature_fraction_seed": 42,
        "bagging_seed": 42,
        "drop_seed": 42,
        "data_random_seed": 42,
        "objective": "rmse",
        "boosting": "gbdt",
        "verbosity": -1,
        "n_jobs": -1,
    }
    pruning_callback = LightGBMPruningCallback(trial, "RMSPE", valid_name="valid_1")

    kf = KFold(n_splits=n_fold, random_state=42, shuffle=True)
    splits = kf.split(X, y)
    lgbm_oof = np.zeros(X.shape[0])

    for fold, (train_idx, valid_idx) in enumerate(splits, 1):
        # create dataset
        X_train, y_train = X.loc[train_idx], y[train_idx]
        X_valid, y_valid = X.loc[valid_idx], y[valid_idx]

        # Root mean squared percentage error weights
        train_weights = 1 / np.square(y_train)
        val_weights = 1 / np.square(y_valid)

        train_dataset = lgbm.Dataset(
            X_train, y_train, weight=train_weights, categorical_feature=["stock_id"]
        )
        val_dataset = lgbm.Dataset(
            X_valid, y_valid, weight=val_weights, categorical_feature=["stock_id"]
        )
        # model
        model = lgbm.train(
            params=params,
            train_set=train_dataset,
            valid_sets=[train_dataset, val_dataset],
            num_boost_round=10000,
            early_stopping_rounds=50,
            feval=feval_RMSPE,
            callbacks=[pruning_callback],
            verbose_eval=False,
        )

        # validation
        lgbm_oof[valid_idx] = model.predict(X_valid, num_iteration=model.best_iteration)

    RMSPE = rmspe(y, lgbm_oof)
    return RMSPE


def group_lgbm_objective(
    trial: FrozenTrial,
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    n_fold: int,
) -> float:
    params = {
        "learning_rate": trial.suggest_float("learning_rate", 1e-02, 2e-01),
        "lambda_l1": trial.suggest_float("lambda_l1", 1, 10),
        "lambda_l2": trial.suggest_float("lambda_l2", 1, 10),
        "num_leaves": trial.suggest_int("num_leaves", 512, 1024),
        "min_sum_hessian_in_leaf": trial.suggest_float(
            "min_sum_hessian_in_leaf", 20, 50
        ),
        "feature_fraction": trial.suggest_uniform("feature_fraction", 0.1, 1),
        "feature_fraction_bynode": trial.suggest_uniform(
            "feature_fraction_bynode", 0.1, 1
        ),
        "bagging_fraction": trial.suggest_uniform("bagging_fraction", 0.1, 1),
        "bagging_freq": trial.suggest_int("bagging_freq", 35, 100),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 512, 1024),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "seed": 42,
        "feature_fraction_seed": 42,
        "bagging_seed": 42,
        "drop_seed": 42,
        "data_random_seed": 42,
        "objective": "rmse",
        "boosting": "gbdt",
        "verbosity": -1,
        "n_jobs": -1,
    }
    pruning_callback = LightGBMPruningCallback(trial, "RMSPE", valid_name="valid_1")

    group_kf = GroupKFold(n_splits=n_fold)
    splits = group_kf.split(X, y, groups=groups)
    lgbm_oof = np.zeros(X.shape[0])

    for fold, (train_idx, valid_idx) in enumerate(splits, 1):
        # create dataset
        X_train, y_train = X.loc[train_idx], y[train_idx]
        X_valid, y_valid = X.loc[valid_idx], y[valid_idx]

        # Root mean squared percentage error weights
        train_weights = 1 / np.square(y_train)
        val_weights = 1 / np.square(y_valid)

        train_dataset = lgbm.Dataset(
            X_train, y_train, weight=train_weights, categorical_feature=["stock_id"]
        )
        val_dataset = lgbm.Dataset(
            X_valid, y_valid, weight=val_weights, categorical_feature=["stock_id"]
        )
        # model
        model = lgbm.train(
            params=params,
            train_set=train_dataset,
            valid_sets=[train_dataset, val_dataset],
            num_boost_round=10000,
            early_stopping_rounds=50,
            feval=feval_RMSPE,
            callbacks=[pruning_callback],
            verbose_eval=False,
        )

        # validation
        lgbm_oof[valid_idx] = model.predict(X_valid, num_iteration=model.best_iteration)

    RMSPE = rmspe(y, lgbm_oof)
    return RMSPE
