import warnings
from typing import Any, Dict, Optional

import keras
import numpy as np
import numpy.matlib
import pandas as pd
from keras.backend import sigmoid
from keras.layers import Activation
from keras.utils.generic_utils import get_custom_objects
from model.model_selection import ShufflableGroupKFold
from pytorch_tabnet.tab_model import TabNetRegressor
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from utils.utils import RMSPE, RMSPELoss, rmspe


def swish(x, beta: int = 1):
    return x * sigmoid(beta * x)


get_custom_objects().update({"swish": Activation(swish)})
warnings.filterwarnings("ignore")

hidden_units = (128, 64, 32)
stock_embedding_size = 24


def base_model():
    # Each instance will consist of two inputs: a single user id, and a single movie id
    stock_id_input = keras.Input(shape=(1,), name="stock_id")
    num_input = keras.Input(shape=(362,), name="num_data")
    # embedding, flatenning and concatenating
    stock_embedded = keras.layers.Embedding(
        max(cat_data) + 1, stock_embedding_size, input_length=1, name="stock_embedding"
    )(stock_id_input)
    stock_flattened = keras.layers.Flatten()(stock_embedded)
    out = keras.layers.Concatenate()([stock_flattened, num_input])

    # Add one or more hidden layers
    for n_hidden in hidden_units:

        out = keras.layers.Dense(n_hidden, activation="swish")(out)

    # out = keras.layers.Concatenate()([out, num_input])

    # A single output: our predicted rating
    out = keras.layers.Dense(1, activation="linear", name="prediction")(out)

    model = keras.Model(
        inputs=[stock_id_input, num_input],
        outputs=out,
    )

    return model


def train_kfold_ffnn(n_fold, train, X_test):
    es = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=20,
        verbose=0,
        mode="min",
        restore_best_weights=True,
    )

    plateau = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.2, patience=7, verbose=0, mode="min"
    )
    model_name = "NN"
    pred_name = "pred_{}".format(model_name)

    n_folds = 5
    kf = ShufflableGroupKFold(n_splits=n_folds, shuffle=True, random_state=42)
    scores_folds[model_name] = []
    counter = 1

    features_to_consider = list(train)

    features_to_consider.remove("time_id")
    features_to_consider.remove("target")
    features_to_consider.remove("row_id")
    try:
        features_to_consider.remove("pred_NN")
    except:
        pass

    train[features_to_consider] = train[features_to_consider].fillna(
        train[features_to_consider].mean()
    )
    test[features_to_consider] = test[features_to_consider].fillna(
        train[features_to_consider].mean()
    )

    train[pred_name] = 0
    test["target"] = 0

    for n_count in range(n_folds):
        print("CV {}/{}".format(counter, n_folds))

        indexes = np.arange(nfolds).astype(int)
        indexes = np.delete(indexes, obj=n_count, axis=0)

        indexes = np.r_[
            values[indexes[0]],
            values[indexes[1]],
            values[indexes[2]],
            values[indexes[3]],
        ]

        X_train = train.loc[train.time_id.isin(indexes), features_to_consider]
        y_train = train.loc[train.time_id.isin(indexes), target_name]
        X_test = train.loc[train.time_id.isin(values[n_count]), features_to_consider]
        y_test = train.loc[train.time_id.isin(values[n_count]), target_name]

        #############################################################################################
        # NN
        #############################################################################################

        model = base_model()

        model.compile(
            keras.optimizers.Adam(learning_rate=0.005), loss=root_mean_squared_per_error
        )

        try:
            features_to_consider.remove("stock_id")
        except:
            pass

        num_data = X_train[features_to_consider]

        scaler = MinMaxScaler(feature_range=(-1, 1))
        num_data = scaler.fit_transform(num_data.values)

        cat_data = X_train["stock_id"]
        target = y_train

        num_data_test = X_test[features_to_consider]
        num_data_test = scaler.transform(num_data_test.values)
        cat_data_test = X_test["stock_id"]

        model.fit(
            [cat_data, num_data],
            target,
            batch_size=1024,
            epochs=1000,
            validation_data=([cat_data_test, num_data_test], y_test),
            callbacks=[es, plateau],
            validation_batch_size=len(y_test),
            shuffle=True,
            verbose=1,
        )
        model.save(f"ffnn_groupkfold{n_count}.h5")
        preds = model.predict([cat_data_test, num_data_test]).reshape(1, -1)[0]

        score = round(rmspe(y_true=y_test, y_pred=preds), 5)
        print("Fold {} {}: {}".format(counter, model_name, score))
        scores_folds[model_name].append(score)

        tt = scaler.transform(test[features_to_consider].values)
        test[target_name] += (
            model.predict([test["stock_id"], tt]).reshape(1, -1)[0].clip(0, 1e10)
        )
        # test[target_name] += model.predict([test['stock_id'], test[features_to_consider]]).reshape(1,-1)[0].clip(0,1e10)

        counter += 1
        features_to_consider.append("stock_id")
