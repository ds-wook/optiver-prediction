import hydra
import numpy as np
import pandas as pd
from data.dataset import network_agg_features
from hydra.utils import to_absolute_path
from model.network import train_kfold_tabnet
from omegaconf import DictConfig
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts


@hydra.main(config_path="../config/train/", config_name="train.yaml")
def _main(cfg: DictConfig):
    path = to_absolute_path(cfg.dataset.path) + "/"
    train = pd.read_pickle(path + cfg.dataset.train)
    test = pd.read_pickle(path + cfg.dataset.test)
    print(train.shape, test.shape)

    # Fill inf values
    train.replace([np.inf, -np.inf], np.nan, inplace=True)
    test.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Aggregating some features
    train, test = network_agg_features(train, test, path)

    # Split features and target
    X = train.drop(["row_id", "target", "time_id"], axis=1)
    y = train["target"]
    X_test = test.drop(["row_id", "time_id"], axis=1)

    # Transform stock id to a numeric value
    X["stock_id"] = X["stock_id"].astype(int)
    X_test["stock_id"] = X_test["stock_id"].astype(int)

    categorical_columns = []
    categorical_dims = {}

    for col in X.columns:
        if col == "stock_id":
            l_enc = LabelEncoder()
            X[col] = l_enc.fit_transform(X[col].values)
            X_test[col] = l_enc.transform(X_test[col].values)
            categorical_columns.append(col)
            categorical_dims[col] = len(l_enc.classes_)
        else:
            scaler = StandardScaler()
            X[col] = scaler.fit_transform(X[col].values.reshape(-1, 1))
            X_test[col] = scaler.transform(X_test[col].values.reshape(-1, 1))

    cat_idxs = [i for i, f in enumerate(X.columns.tolist()) if f in categorical_columns]

    cat_dims = [
        categorical_dims[f]
        for i, f in enumerate(X.columns.tolist())
        if f in categorical_columns
    ]

    params = dict(
        cat_idxs=cat_idxs,
        cat_dims=cat_dims,
        cat_emb_dim=1,
        n_d=16,
        n_a=16,
        n_steps=2,
        gamma=2,
        n_independent=2,
        n_shared=2,
        lambda_sparse=0,
        optimizer_fn=Adam,
        optimizer_params=dict(lr=(2e-2)),
        mask_type="entmax",
        scheduler_params=dict(
            T_0=200, T_mult=1, eta_min=1e-4, last_epoch=-1, verbose=False
        ),
        scheduler_fn=CosineAnnealingWarmRestarts,
        seed=42,
        verbose=10,
    )
    tabnet_preds = train_kfold_tabnet(cfg.model.fold, X, y, X_test, params)
    # Save test predictions
    test["target"] = tabnet_preds
    test[["row_id", "target"]].to_csv("submission.csv", index=False)


if __name__ == "__main__":
    _main()
