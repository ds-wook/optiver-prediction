import gc
from abc import abstractclassmethod
from typing import Any, Callable, Dict, NamedTuple, Optional, Union

import joblib
import numpy as np
import pandas as pd
from hydra.utils import to_absolute_path
from tqdm import tqdm

from model.model_selection import ShufflableGroupKFold


class ModelResult(NamedTuple):
    oof_preds: np.ndarray
    preds: Optional[np.ndarray]
    models: Dict[str, Any]
    scores: Dict[str, float]


class BaseModel:
    def __init__(self, fold: int, metric: Callable):
        self.fold = fold
        self.metric = metric
        self.result = None

    @abstractclassmethod
    def _train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_valid: pd.DataFrame,
        y_valid: pd.Series,
        fold: int,
        is_search: Union[bool] = False,
        verbose: Union[bool] = False,
    ):
        raise NotImplementedError

    def train(
        self,
        train_x: pd.DataFrame,
        train_y: pd.Series,
        groups: pd.Series,
        verbose: Union[bool] = False,
    ) -> ModelResult:
        """
        Train data
            Parameter:
                train_x: train dataset
                train_y: target dataset
                groups: group fold parameters
                params: lightgbm' parameters
                verbose: log lightgbm' training
            Return:
                True: Finish Training
        """

        models = dict()
        scores = dict()

        str_kf = ShufflableGroupKFold(n_splits=self.fold, shuffle=True, random_state=42)
        splits = str_kf.split(train_x, train_y, groups)

        oof_preds = np.zeros(train_x.shape[0])

        for fold, (train_idx, valid_idx) in enumerate(splits, 1):
            X_train, y_train = train_x.iloc[train_idx], train_y.iloc[train_idx]
            X_valid, y_valid = train_x.iloc[valid_idx], train_y.iloc[valid_idx]

            # model
            model = self._train(
                X_train,
                y_train,
                X_valid,
                y_valid,
                fold=fold,
                verbose=verbose,
            )
            models[f"fold_{fold}"] = model

            # validation
            oof_preds[valid_idx] = model.predict(
                X_valid, num_iteration=model.best_iteration_
            )

            score = self.metric(y_valid.values, oof_preds[valid_idx])
            scores[f"fold_{fold}"] = score
            gc.collect()

            del X_train, X_valid, y_train, y_valid

        oof_score = self.metric(train_y.values, oof_preds)

        self.result = ModelResult(
            oof_preds=oof_preds,
            models=models,
            preds=None,
            scores={"oof_score": oof_score, "KFold_scores": scores},
        )
        model_path = to_absolute_path(f"../models/lightgbm/lgbm_group_kfold{fold}.pkl")
        # save model
        joblib.dump(model, model_path)

        return self.result

    def predict(self, test_x: pd.DataFrame) -> np.ndarray:
        """
        Predict data
            Parameter:
                test_x: test dataset
            Return:
                preds: inference prediction
        """
        folds = self.fold
        preds = np.zeros(test_x.shape[0])

        for fold in tqdm(range(1, folds + 1)):
            model = self.result.models[f"fold_{fold}"]
            preds += model.predict(test_x, num_iteration=model.best_iteration_) / folds

        assert len(preds) == len(test_x)

        return preds
