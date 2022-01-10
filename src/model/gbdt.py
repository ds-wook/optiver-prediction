import warnings
from typing import Any, Dict, Optional, Union

import neptune.new as neptune
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from neptune.new.integrations.lightgbm import NeptuneCallback, create_booster_summary

from model.base import BaseModel
from utils.utils import feval_metric

warnings.filterwarnings("ignore")


class LightGBMTrainer(BaseModel):
    def __init__(
        self,
        params: Optional[Dict[str, Any]],
        run: Optional[neptune.init],
        search: bool = False,
        **kwargs,
    ):
        self.params = params
        self.run = run
        self.search = search
        super().__init__(**kwargs)

    def _get_default_params(self) -> Dict[str, Any]:
        """
        setting default parameters
        Return:
            LightGBM default parameter
        """

        return {
            "n_estimators": 10000,
            "boosting_type": "gbdt",
            "objective": "rmse",
            "random_state": 42,
            "learning_rate": 0.05,
            "num_leaves": 5,
            "max_bin": 55,
            "subsample": 0.8,
            "min_child_sample": 6,
            "min_child_weight": 11,
        }

    def _train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_valid: pd.DataFrame,
        y_valid: pd.Series,
        fold: int,
        is_search: Union[bool] = False,
        verbose: Union[bool] = False,
    ) -> LGBMRegressor:
        """method train"""
        # Root mean squared percentage error weights
        train_weights = 1 / np.square(y_train)
        val_weights = 1 / np.square(y_valid)

        neptune_callback = (
            NeptuneCallback(run=self.run, base_namespace=f"fold_{fold}")
            if is_search is self.search
            else self.run
        )

        model = (
            LGBMRegressor(**self.params)
            if self.params is not None
            else LGBMRegressor(**self._get_default_params())
        )

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

        if is_search is self.search:
            # Log summary metadata to the same run under the "lgbm_summary" namespace
            self.run[f"lgbm_summary/fold_{fold}"] = create_booster_summary(
                booster=model,
                y_pred=model.predict(X_valid),
                y_true=y_valid,
            )

        return model
