from typing import Tuple, Union

import lightgbm as lgbm
import numpy as np


def rmspe(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.sqrt(np.mean(np.square((y_true - y_pred) / y_true)))


def feval_RMSPE(
    preds: np.ndarray, lgbm_train: lgbm.Dataset
) -> Tuple[Union[str, float, bool]]:
    labels = lgbm_train.get_label()
    return "RMSPE", rmspe(y_true=labels, y_pred=preds), False
