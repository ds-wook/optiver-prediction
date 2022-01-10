import warnings
from typing import Any, Dict, Optional, Tuple, Union

import joblib
import neptune.new as neptune
import numpy as np
import pandas as pd
from hydra.utils import to_absolute_path
from lightgbm import LGBMRegressor
from model.model_selection import ShufflableGroupKFold
from neptune.new.integrations.lightgbm import NeptuneCallback, create_booster_summary
from utils.utils import feval_metric, rmspe

warnings.filterwarnings("ignore")


def run_kfold_lightgbm(
    n_fold: int,
    X: pd.DataFrame,
    y: pd.DataFrame,
    X_test: pd.DataFrame,
    groups: pd.Series,
    params: Optional[Dict[str, Any]] = None,
    verbose: Union[int, bool] = False,
) -> Tuple[np.ndarray, np.ndarray]:

    kf = ShufflableGroupKFold(n_splits=n_fold, random_state=42, shuffle=True)
    splits = kf.split(X, y, groups)
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
        model_path = to_absolute_path(
            f"../../models/lgbm_model/lgbm_group_kfold{fold}.pkl"
        )
        # save model
        joblib.dump(model, model_path)

        # Log summary metadata to the same run under the "lgbm_summary" namespace
        run[f"lgbm_summary/fold_{fold}"] = create_booster_summary(
            booster=model,
            log_trees=True,
            list_trees=[0, 1, 2, 3, 4],
            max_num_features=20,
            y_pred=lgb_oof[valid_idx],
            y_true=y_valid,
        )

    print(f"Total Performance RMSPE: {rmspe(y, lgb_oof)}")
    run.stop()

    return lgb_preds


def load_lightgbm_model(
    n_fold: int,
    X: pd.DataFrame,
    y: pd.DataFrame,
    X_test: pd.DataFrame,
    groups: Optional[pd.Series] = None,
) -> Any:
    group_kf = ShufflableGroupKFold(n_splits=n_fold, random_state=42, shuffle=True)
    splits = group_kf.split(X=X, y=y, groups=groups)
    lgb_oof = np.zeros(X.shape[0])
    lgb_preds = np.zeros(X_test.shape[0])
    for fold, (train_idx, valid_idx) in enumerate(splits, 1):
        # create dataset
        X_valid, y_valid = X.iloc[valid_idx], y.iloc[valid_idx]

        # model
        path = to_absolute_path(f"../../models/lgbm_model/lgbm_group_kfold{fold}.pkl")
        model = joblib.load(path)

        # validation
        lgb_oof[valid_idx] = model.predict(X_valid, num_iteration=model.best_iteration_)
        lgb_preds += model.predict(X_test, num_iteration=model.best_iteration_) / n_fold
        RMSPE = rmspe(y_true=y_valid, y_pred=lgb_oof[valid_idx])
        print(f"Performance of the　prediction: , RMSPE: {RMSPE}")

    print(f"Total Performance RMSPE: {rmspe(y, lgb_oof)}")
    return lgb_preds
