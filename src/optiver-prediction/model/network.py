import warnings
from typing import Any, Dict, Optional

import numpy as np
import numpy.matlib
import pandas as pd
from pytorch_tabnet.tab_model import TabNetRegressor
from sklearn.model_selection import KFold
from utils.utils import RMSPE, RMSPELoss, rmspe

warnings.filterwarnings("ignore")


def train_kfold_tabnet(
    n_fold: int,
    X: pd.DataFrame,
    y: pd.DataFrame,
    X_test: pd.DataFrame,
    params: Optional[Dict[str, Any]] = None,
) -> np.ndarray:

    kfold = KFold(n_splits=5, random_state=42, shuffle=True)
    # Create out of folds array
    oof_predictions = np.zeros((X.shape[0], 1))
    test_predictions = np.zeros(X_test.shape[0])

    for fold, (trn_ind, val_ind) in enumerate(kfold.split(X)):
        print(f"Training fold {fold + 1}")
        X_train, X_val = X.iloc[trn_ind].values, X.iloc[val_ind].values
        y_train, y_val = y.iloc[trn_ind].values.reshape(-1, 1), y.iloc[
            val_ind
        ].values.reshape(-1, 1)

        model = TabNetRegressor(**params)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            max_epochs=200,
            patience=50,
            batch_size=1024 * 20,
            virtual_batch_size=128 * 20,
            num_workers=4,
            drop_last=False,
            eval_metric=[RMSPE],
            loss_fn=RMSPELoss,
        )

        oof_predictions[val_ind] = model.predict(X_val)
        test_predictions += model.predict(X_test.values).flatten() / n_fold

    print(f"OOF score across folds: {rmspe(y, oof_predictions.flatten())}")
    return test_predictions
