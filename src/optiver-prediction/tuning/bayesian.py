import warnings
from typing import Callable, Sequence, Union

import neptune.new as neptune
import neptune.new.integrations.optuna as optuna_utils
import numpy as np
import optuna
import pandas as pd
import yaml
from hydra.utils import to_absolute_path
from lightgbm import LGBMRegressor
from neptune.new.exceptions import NeptuneMissingApiTokenException
from optuna.integration import LightGBMPruningCallback
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from optuna.study import Study
from optuna.trial import FrozenTrial, Trial
from utils.utils import feval_metric, rmspe
from model.model_selection import ShufflableGroupKFold
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
        params["n_estimators"] = 10000
        params["boosting_type"] = "gbdt"
        params["objective"] = "rmse"
        params["verbosity"] = -1
        params["n_jobs"] = -1
        params["seed"] = 42
        params["drop_seed"] = 42

        with open(to_absolute_path("../../config/train/train.yaml")) as f:
            train_dict = yaml.load(f, Loader=yaml.FullLoader)
        train_dict["params"] = params

        with open(to_absolute_path("../../config/train/" + params_name), "w") as p:
            yaml.dump(train_dict, p)


def lgbm_objective(
    trial: FrozenTrial,
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    n_fold: int,
) -> float:
    params = {
        "learning_rate": trial.suggest_float("learning_rate", 1e-02, 2e-01),
        "reg_alpha": trial.suggest_float("reg_alpha", 1, 10),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 2),
        "num_leaves": trial.suggest_int("num_leaves", 512, 1024),
        "min_child_weight": trial.suggest_float("min_child_weight", 20, 50),
        "colsample_bytree": trial.suggest_uniform("colsample_bytree", 0.1, 1),
        "min_split_gain": trial.suggest_uniform("min_split_gain", 0.1, 1),
        "subsample": trial.suggest_uniform("subsample", 0.1, 1),
        "subsample_freq": trial.suggest_int("subsample_freq", 35, 100),
        "min_child_samples": trial.suggest_int("min_child_samples", 512, 1024),
        "max_depth": trial.suggest_int("max_depth", 3, 15),
        "n_estimators": 10000,
        "objective": "rmse",
        "boosting_type": "gbdt",
        "drop_seed": 42,
        "seed": 42,
        "verbosity": -1,
        "n_jobs": -1,
    }
    pruning_callback = LightGBMPruningCallback(trial, "RMSPE", valid_name="valid_1")

    group_kf = ShufflableGroupKFold(n_splits=n_fold, random_state=42, shuffle=True)
    splits = group_kf.split(X, y, groups)
    lgbm_oof = np.zeros(X.shape[0])

    for fold, (train_idx, valid_idx) in enumerate(splits, 1):
        # create dataset
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_valid, y_valid = X.iloc[valid_idx], y.iloc[valid_idx]

        # Root mean squared percentage error weights
        train_weights = 1 / np.square(y_train)
        val_weights = 1 / np.square(y_valid)

        # model
        model = LGBMRegressor(**params)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_train, y_train), (X_valid, y_valid)],
            eval_metric=feval_metric,
            sample_weight=train_weights,
            eval_sample_weight=[val_weights],
            early_stopping_rounds=50,
            verbose=False,
            categorical_feature=["stock_id"],
            callbacks=[pruning_callback],
        )
        # validation
        lgbm_oof[valid_idx] = model.predict(
            X_valid, num_iteration=model.best_iteration_
        )

    RMSPE = rmspe(y, lgbm_oof)
    return RMSPE
