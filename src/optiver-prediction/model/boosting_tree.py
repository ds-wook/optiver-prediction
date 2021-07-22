import warnings
from typing import Any, Dict, Optional, Tuple, Union

import lightgbm as lgbm
import neptune.new as neptune
import numpy as np
import pandas as pd
from neptune.new.integrations.lightgbm import NeptuneCallback, create_booster_summary
from sklearn.model_selection import KFold
from utils.utils import feval_RMSPE, rmspe

warnings.filterwarnings("ignore")


def run_kfold_lightgbm(
    n_fold: int,
    X: pd.DataFrame,
    y: pd.DataFrame,
    X_test: pd.DataFrame,
    params: Optional[Dict[str, Any]] = None,
    token: Optional[str] = None,
    verbose: Union[int, bool] = False,
) -> Tuple[np.ndarray, np.ndarray]:
    kf = KFold(n_splits=n_fold, random_state=42, shuffle=True)
    splits = kf.split(X, y)
    lgb_oof = np.zeros(X.shape[0])
    lgb_preds = np.zeros(X_test.shape[0])

    for fold, (train_idx, valid_idx) in enumerate(splits, 1):

        print("Fold :", fold)

        # create dataset
        X_train, y_train = X.loc[train_idx], y[train_idx]
        X_valid, y_valid = X.loc[valid_idx], y[valid_idx]

        # RMSPE weight
        weights = 1 / np.square(y_train)
        lgbm_train = lgbm.Dataset(X_train, y_train, weight=weights)

        weights = 1 / np.square(y_valid)
        lgbm_valid = lgbm.Dataset(
            X_valid, y_valid, reference=lgbm_train, weight=weights
        )
        try:
            run = neptune.init(
                project="ds-wook/optiver-prediction",
                api_token=token,
            )
            neptune_callback = NeptuneCallback(run=run)
            # model
            model = lgbm.train(
                params=params,
                train_set=lgbm_train,
                valid_sets=[lgbm_train, lgbm_valid],
                num_boost_round=5000,
                feval=feval_RMSPE,
                verbose_eval=100,
                categorical_feature=["stock_id"],
                callbacks=[neptune_callback],
            )

            # validation
            lgb_oof[valid_idx] = model.predict(
                X_valid, num_iteration=model.best_iteration
            )
            lgb_preds += (
                model.predict(X_test, num_iteration=model.best_iteration) / n_fold
            )
            RMSPE = round(rmspe(y_true=y_valid, y_pred=lgb_oof[valid_idx]), 3)

            print(f"Performance of the　prediction: , RMSPE: {RMSPE}")
            # Log summary metadata to the same run under the "lgbm_summary" namespace
            run["lgbm_summary"] = create_booster_summary(
                booster=model,
                log_trees=True,
                list_trees=[0, 1, 2, 3, 4],
                y_pred=lgb_oof[valid_idx],
                y_true=y_valid,
            )
            run.stop()

        except TypeError:
            # model
            model = lgbm.train(
                params=params,
                train_set=lgbm_train,
                valid_sets=[lgbm_train, lgbm_valid],
                num_boost_round=5000,
                feval=feval_RMSPE,
                verbose_eval=100,
                categorical_feature=["stock_id"],
            )

            # validation
            y_pred = model.predict(X_valid, num_iteration=model.best_iteration)

            RMSPE = round(rmspe(y_true=y_valid, y_pred=y_pred), 3)
            print(f"Performance of the　prediction: , RMSPE: {RMSPE}")

    return lgb_oof, lgb_preds
