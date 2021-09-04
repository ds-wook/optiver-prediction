import warnings
from typing import Any, Dict, Optional, Tuple, Union

import joblib
import lightgbm as lgbm
import neptune.new as neptune
import numpy as np
import pandas as pd
from hydra.utils import to_absolute_path
from lightgbm import LGBMRegressor
from neptune.new.integrations import xgboost
from neptune.new.integrations.lightgbm import NeptuneCallback, create_booster_summary
from sklearn.model_selection import GroupKFold, KFold
from utils.utils import feval_metric, feval_RMSPE, feval_rmspe, rmspe
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")


def run_kfold_lightgbm(
    n_fold: int,
    X: pd.DataFrame,
    y: pd.DataFrame,
    X_test: pd.DataFrame,
    params: Optional[Dict[str, Any]] = None,
    verbose: Union[int, bool] = False,
) -> Tuple[np.ndarray, np.ndarray]:

    kf = KFold(n_splits=n_fold, random_state=2021, shuffle=True)
    splits = kf.split(X)
    lgb_oof = np.zeros(X.shape[0])
    lgb_preds = np.zeros(X_test.shape[0])

    run = neptune.init(project="ds-wook/optiver-prediction", tags=["LightGBM", "KFold"])

    for fold, (train_idx, valid_idx) in enumerate(splits, 1):
        print("Fold :", fold)
        neptune_callback = NeptuneCallback(run=run, base_namespace=f"fold_{fold}")
        # create dataset
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_valid, y_valid = X.iloc[valid_idx], y.iloc[valid_idx]

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
            verbose_eval=250,
            feval=feval_RMSPE,
            callbacks=[neptune_callback],
        )

        # validation
        lgb_oof[valid_idx] = model.predict(X_valid, num_iteration=model.best_iteration)
        lgb_preds += model.predict(X_test, num_iteration=model.best_iteration) / n_fold
        RMSPE = rmspe(y_true=y_valid, y_pred=lgb_oof[valid_idx])

        print(f"Performance of the　prediction: , RMSPE: {RMSPE}")

        model_path = to_absolute_path(
            f"../../models/lgbm_model/best_lgbm_kfold{fold}.txt"
        )
        # save model
        model.save_model(model_path, num_iteration=model.best_iteration)

        # Log summary metadata to the same run under the "lgbm_summary" namespace
        run[f"lgbm_summary/fold_{fold}"] = create_booster_summary(
            booster=model,
            log_trees=True,
            list_trees=[0, 1, 2, 3, 4],
            y_pred=lgb_oof[valid_idx],
            y_true=y_valid,
        )

    print(f"Total Performance RMSPE: {rmspe(y, lgb_oof)}")
    run.stop()

    return lgb_preds


def run_kfold_xgboost(
    n_fold: int,
    X: pd.DataFrame,
    y: pd.DataFrame,
    params: Optional[Dict[str, Any]] = None,
    verbose: Union[int, bool] = False,
) -> Tuple[np.ndarray, np.ndarray]:

    kf = KFold(n_splits=n_fold, random_state=42, shuffle=True)
    splits = kf.split(X)
    xgb_oof = np.zeros(X.shape[0])

    run = neptune.init(project="ds-wook/optiver-prediction", tags=["XGBoost", "KFold"])

    for fold, (train_idx, valid_idx) in enumerate(splits, 1):

        print("Fold :", fold)

        # create dataset
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_valid, y_valid = X.iloc[valid_idx], y.iloc[valid_idx]

        # Root mean squared percentage error weights
        train_weights = 1 / np.square(y_train)
        val_weights = 1 / np.square(y_valid)

        neptune_callback = xgboost.NeptuneCallback(
            run=run,
            base_namespace=f"fold_{fold}",
            log_tree=[0, 1, 2, 3],
            max_num_features=10,
        )
        # model
        model = XGBRegressor(**params)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_train, y_train), (X_valid, y_valid)],
            early_stopping_rounds=10,
            sample_weight=train_weights,
            sample_weight_eval_set=[train_weights, val_weights],
            eval_metric=feval_rmspe,
            callbacks=[neptune_callback],
            verbose=verbose,
        )
        # validation
        xgb_oof[valid_idx] = model.predict(X_valid)
        RMSPE = rmspe(y_true=y_valid, y_pred=xgb_oof[valid_idx])

        print(f"Performance of the　prediction: , RMSPE: {RMSPE}")

        model_path = to_absolute_path(
            f"../../models/xgb_model/best_xgb_kfold{fold}.pkl"
        )
        # save model
        joblib.dump(model, model_path)

    run.stop()

    return xgb_oof


def run_group_kfold_lightgbm(
    n_fold: int,
    X: pd.DataFrame,
    y: pd.DataFrame,
    X_test: pd.DataFrame,
    groups: pd.Series,
    params: Optional[Dict[str, Any]] = None,
    verbose: Union[int, bool] = False,
) -> Tuple[np.ndarray, np.ndarray]:

    group_kf = GroupKFold(n_splits=n_fold)
    splits = group_kf.split(X=X, y=y, groups=groups)
    lgb_oof = np.zeros(X.shape[0])
    lgb_preds = np.zeros(X_test.shape[0])

    run = neptune.init(project="ds-wook/optiver-prediction", tags=["LightGBM", "Group"])

    for fold, (train_idx, valid_idx) in enumerate(splits, 1):
        print("Fold :", fold)
        neptune_callback = NeptuneCallback(run=run, base_namespace=f"fold_{fold}")
        # create dataset
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_valid, y_valid = X.iloc[valid_idx], y.iloc[valid_idx]

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
            verbose_eval=50,
            callbacks=[neptune_callback],
        )

        # validation
        lgb_oof[valid_idx] = model.predict(X_valid, num_iteration=model.best_iteration)
        lgb_preds += model.predict(X_test, num_iteration=model.best_iteration) / n_fold
        RMSPE = round(rmspe(y_true=y_valid, y_pred=lgb_oof[valid_idx]), 3)

        print(f"Performance of the　prediction: , RMSPE: {RMSPE}")

        model_path = to_absolute_path(
            f"../../models/lgbm_model/lgbm_groupfold{fold}.txt"
        )
        # save model
        model.save_model(model_path, num_iteration=model.best_iteration)

        # Log summary metadata to the same run under the "lgbm_summary" namespace
        run[f"lgbm_summary/fold_{fold}"] = create_booster_summary(
            booster=model,
            log_trees=True,
            list_trees=[0, 1, 2, 3, 4],
            y_pred=lgb_oof[valid_idx],
            y_true=y_valid,
        )

    print(f"Total Performance RMSPE: {rmspe(y, lgb_oof)}")
    run.stop()
    return lgb_preds


def train_group_kfold_lightgbm(
    n_fold: int,
    X: pd.DataFrame,
    y: pd.DataFrame,
    X_test: pd.DataFrame,
    groups: pd.Series,
    params: Optional[Dict[str, Any]] = None,
    verbose: Union[int, bool] = False,
) -> Tuple[np.ndarray, np.ndarray]:

    group_kf = GroupKFold(n_splits=n_fold)
    splits = group_kf.split(X=X, y=y, groups=groups)
    lgb_oof = np.zeros(X.shape[0])
    lgb_preds = np.zeros(X_test.shape[0])

    run = neptune.init(project="ds-wook/optiver-prediction", tags=["LightGBM", "Group"])

    for fold, (train_idx, valid_idx) in enumerate(splits, 1):
        print("Fold :", fold)
        neptune_callback = NeptuneCallback(run=run, base_namespace=f"fold_{fold}")
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
            verbose=verbose,
            categorical_feature=["stock_id"],
            callbacks=[neptune_callback],
        )

        # validation
        lgb_oof[valid_idx] = model.predict(X_valid, num_iteration=model.best_iteration_)
        lgb_preds += model.predict(X_test, num_iteration=model.best_iteration_) / n_fold
        RMSPE = rmspe(y_true=y_valid, y_pred=lgb_oof[valid_idx])

        print(f"Performance of the　prediction: , RMSPE: {RMSPE}")

        model_path = to_absolute_path(
            f"../../models/lgbm_model/sklearn_lgbm_group_kfold{fold}.pkl"
        )
        # save model
        joblib.dump(model, model_path)

        # Log summary metadata to the same run under the "lgbm_summary" namespace
        run[f"lgbm_summary/fold_{fold}"] = create_booster_summary(
            booster=model,
            log_trees=True,
            list_trees=[0, 1, 2, 3, 4],
            y_pred=lgb_oof[valid_idx],
            y_true=y_valid,
        )

    print(f"Total Performance RMSPE: {rmspe(y, lgb_oof)}")
    run.stop()
    return lgb_preds


def run_group_kfold_xgboost(
    n_fold: int,
    X: pd.DataFrame,
    y: pd.DataFrame,
    groups: pd.Series,
    params: Optional[Dict[str, Any]] = None,
    verbose: Union[int, bool] = False,
) -> Tuple[np.ndarray, np.ndarray]:

    group_kf = GroupKFold(n_splits=n_fold)
    splits = group_kf.split(X=X, y=y, groups=groups)
    xgb_oof = np.zeros(X.shape[0])
    run = neptune.init(project="ds-wook/optiver-prediction", tags=["XGBoost", "Group"])
    for fold, (train_idx, valid_idx) in enumerate(splits, 1):

        print("Fold :", fold)

        # create dataset
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_valid, y_valid = X.iloc[valid_idx], y.iloc[valid_idx]

        # Root mean squared percentage error weights
        train_weights = 1 / np.square(y_train)
        val_weights = 1 / np.square(y_valid)

        neptune_callback = xgboost.NeptuneCallback(
            run=run,
            base_namespace=f"fold_{fold}",
            log_tree=[0, 1, 2, 3],
            max_num_features=10,
        )
        # model
        model = XGBRegressor(**params)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_train, y_train), (X_valid, y_valid)],
            early_stopping_rounds=10,
            sample_weight=train_weights,
            sample_weight_eval_set=[train_weights, val_weights],
            eval_metric=feval_rmspe,
            callbacks=[neptune_callback],
            verbose=verbose,
        )
        # validation
        xgb_oof[valid_idx] = model.predict(X_valid)
        RMSPE = round(rmspe(y_true=y_valid, y_pred=xgb_oof[valid_idx]), 3)

        print(f"Performance of the　prediction: , RMSPE: {RMSPE}")

        model_path = to_absolute_path(
            f"../../models/xgb_model/best_xgb_groupfold{fold}.pkl"
        )
        # save model
        joblib.dump(model, model_path)

    run.stop()

    return xgb_oof
