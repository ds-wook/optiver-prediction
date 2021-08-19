from typing import Tuple, Union

import lightgbm as lgbm
import numpy as np
import xgboost as xgb


def rmspe(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.sqrt(np.mean(np.square((y_true - y_pred) / y_true)))


def feval_RMSPE(
    preds: np.ndarray, lgbm_train: lgbm.Dataset
) -> Tuple[Union[str, float, bool]]:
    labels = lgbm_train.get_label()
    return "RMSPE", rmspe(y_true=labels, y_pred=preds), False


def gradient(predt: np.ndarray, dtrain: xgb.DMatrix) -> np.ndarray:
    """Compute the gradient squared percentage error."""
    y = dtrain.get_label()
    return -2 * (y - predt) / (y ** 2)


def hessian(predt: np.ndarray, dtrain: xgb.DMatrix) -> np.ndarray:
    """Compute the hessian for squared percentage error."""
    y = dtrain.get_label()
    return 2 / (y ** 2)


def squared_percentage(
    predt: np.ndarray, dtrain: xgb.DMatrix
) -> Tuple[np.ndarray, np.ndarray]:
    """Squared Percentage Error objective."""
    grad = gradient(predt, dtrain)
    hess = hessian(predt, dtrain)
    return grad, hess


def feval_rmspe(predt: np.ndarray, dtrain: xgb.DMatrix) -> Tuple[str, np.ndarray]:
    y = dtrain.get_label()
    return "RMSPE", np.sqrt(np.mean(np.square((y - predt) / y)))
