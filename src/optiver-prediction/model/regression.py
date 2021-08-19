import warnings
from typing import Any, Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from hydra.utils import to_absolute_path
from rgf.sklearn import RGFRegressor
from sklearn.model_selection import KFold
from utils.utils import rmspe

warnings.filterwarnings("ignore")


def run_kfold_rgf(
    n_fold: int,
    X: pd.DataFrame,
    y: pd.DataFrame,
    params: Optional[Dict[str, Any]] = None,
) -> Tuple[np.ndarray, np.ndarray]:

    kf = KFold(n_splits=n_fold, random_state=42, shuffle=True)
    splits = kf.split(X)
    rgf_oof = np.zeros(X.shape[0])

    for fold, (train_idx, valid_idx) in enumerate(splits, 1):
        print("Fold :", fold)

        # create dataset
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_valid, y_valid = X.iloc[valid_idx], y.iloc[valid_idx]

        # Root mean squared percentage error weights
        train_weights = 1 / np.square(y_train)

        # model
        model = RGFRegressor(**params)
        model.fit(X=X_train, y=y_train, sample_weight=train_weights)

        # validation
        rgf_oof[valid_idx] = model.predict(X_valid)

        RMSPE = round(rmspe(y_true=y_valid, y_pred=rgf_oof[valid_idx]), 3)
        print(f"Performance of the　prediction: , RMSPE: {RMSPE}")
    model_path = to_absolute_path(f"../../models/rgf_model/rgf_kfold{fold}.pkl")
    # save model
    joblib.dump(model, model_path)
    return rgf_oof
