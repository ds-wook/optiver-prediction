from typing import Tuple, Union

import lightgbm as lgbm
import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from pytorch_tabnet.metrics import Metric


class RMSPE(Metric):
    def __init__(self):
        self._name = "rmspe"
        self._maximize = False

    def __call__(self, y_true, y_score):
        return np.sqrt(np.mean(np.square((y_true - y_score) / y_true)))


def RMSPELoss(y_pred, y_true):
    return torch.sqrt(torch.mean(((y_true - y_pred) / y_true) ** 2)).clone()


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


def feval_metric(
    y_true: np.ndarray, y_pred: np.ndarray
) -> Tuple[Union[str, float, bool]]:
    return "RMSPE", rmspe(y_true=y_true, y_pred=y_pred), False


def reduce_mem_usage(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    numerics = ["int16", "int32", "int64", "float16", "float32", "float64"]
    start_mem = df.memory_usage().sum() / 1024 ** 2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == "int":
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if (
                    c_min > np.finfo(np.float16).min
                    and c_max < np.finfo(np.float16).max
                ):
                    df[col] = df[col].astype(np.float16)
                elif (
                    c_min > np.finfo(np.float32).min
                    and c_max < np.finfo(np.float32).max
                ):
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024 ** 2
    if verbose:
        print(
            "Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)".format(
                end_mem, 100 * (start_mem - end_mem) / start_mem
            )
        )
    return df
