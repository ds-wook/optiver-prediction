from typing import List, Tuple

import numpy as np
import pandas as pd

data_dir = "../input/optiver-realized-volatility-prediction/"


def calc_wap(df: pd.DataFrame) -> pd.Series:
    wap = (df["bid_price1"] * df["ask_size1"] + df["ask_price1"] * df["bid_size1"]) / (
        df["bid_size1"] + df["ask_size1"]
    )
    return wap


def calc_wap2(df: pd.DataFrame) -> pd.Series:
    wap = (df["bid_price2"] * df["ask_size2"] + df["ask_price2"] * df["bid_size2"]) / (
        df["bid_size2"] + df["ask_size2"]
    )
    return wap


def calc_wap3(df: pd.DataFrame) -> pd.Series:
    wap = (df["bid_price2"] * df["bid_size2"] + df["ask_price2"] * df["ask_size2"]) / (
        df["bid_size2"] + df["ask_size2"]
    )
    return wap


def log_return(list_stock_prices: List[float]) -> float:
    return np.log(list_stock_prices).diff()


def realized_volatility(series: pd.Series) -> float:
    return np.sqrt(np.sum(series ** 2))


def count_unique(series: pd.Series) -> int:
    return len(np.unique(series))


def preprocessor_book(file_path: str) -> pd.DataFrame:
    df = pd.read_parquet(file_path)
    # calculate return etc
    df["wap"] = calc_wap(df)
    df["log_return"] = df.groupby("time_id")["wap"].apply(log_return)

    df["wap2"] = calc_wap2(df)
    df["log_return2"] = df.groupby("time_id")["wap2"].apply(log_return)

    df["wap3"] = calc_wap3(df)
    df["log_return3"] = df.groupby("time_id")["wap3"].apply(log_return)

    df["wap_balance"] = abs(df["wap"] - df["wap2"])

    df["price_spread"] = (df["ask_price1"] - df["bid_price1"]) / (
        (df["ask_price1"] + df["bid_price1"]) / 2
    )
    df["bid_spread"] = df["bid_price1"] - df["bid_price2"]
    df["ask_spread"] = df["ask_price1"] - df["ask_price2"]
    df["total_volume"] = (df["ask_size1"] + df["ask_size2"]) + (
        df["bid_size1"] + df["bid_size2"]
    )
    df["volume_imbalance"] = abs(
        (df["ask_size1"] + df["ask_size2"]) - (df["bid_size1"] + df["bid_size2"])
    )

    # dict for aggregate
    create_feature_dict = {
        "log_return": [realized_volatility],
        "log_return2": [realized_volatility],
        "log_return3": [realized_volatility],
        "wap_balance": [np.mean],
        "price_spread": [np.mean],
        "bid_spread": [np.mean],
        "ask_spread": [np.mean],
        "volume_imbalance": [np.mean],
        "total_volume": [np.mean],
        "wap": [np.mean],
    }

    # groupby / all seconds
    df_feature = pd.DataFrame(
        df.groupby(["time_id"]).agg(create_feature_dict)
    ).reset_index()

    df_feature.columns = [
        "_".join(col) for col in df_feature.columns
    ]  # time_id is changed to time_id_

    # groupby / last XX seconds
    last_seconds = [300]

    for second in last_seconds:
        second = 600 - second

        df_feature_sec = pd.DataFrame(
            df.query(f"seconds_in_bucket >= {second}")
            .groupby(["time_id"])
            .agg(create_feature_dict)
        ).reset_index()

        df_feature_sec.columns = [
            "_".join(col) for col in df_feature_sec.columns
        ]  # time_id is changed to time_id_

        df_feature_sec = df_feature_sec.add_suffix("_" + str(second))

        df_feature = pd.merge(
            df_feature,
            df_feature_sec,
            how="left",
            left_on="time_id_",
            right_on=f"time_id__{second}",
        )
        df_feature = df_feature.drop([f"time_id__{second}"], axis=1)

    # create row_id
    stock_id = file_path.split("=")[1]
    df_feature["row_id"] = df_feature["time_id_"].apply(lambda x: f"{stock_id}-{x}")
    df_feature = df_feature.drop(["time_id_"], axis=1)

    return df_feature


def preprocessor_trade(file_path: str) -> pd.DataFrame:
    df = pd.read_parquet(file_path)
    df["log_return"] = df.groupby("time_id")["price"].apply(log_return)

    aggregate_dictionary = {
        "log_return": [realized_volatility],
        "seconds_in_bucket": [count_unique],
        "size": [np.sum],
        "order_count": [np.mean],
    }

    df_feature = df.groupby("time_id").agg(aggregate_dictionary)

    df_feature = df_feature.reset_index()
    df_feature.columns = ["_".join(col) for col in df_feature.columns]

    # groupby / last XX seconds
    last_seconds = [300]

    for second in last_seconds:
        second = 600 - second

        df_feature_sec = (
            df.query(f"seconds_in_bucket >= {second}")
            .groupby("time_id")
            .agg(aggregate_dictionary)
        )
        df_feature_sec = df_feature_sec.reset_index()

        df_feature_sec.columns = ["_".join(col) for col in df_feature_sec.columns]
        df_feature_sec = df_feature_sec.add_suffix("_" + str(second))

        df_feature = pd.merge(
            df_feature,
            df_feature_sec,
            how="left",
            left_on="time_id_",
            right_on=f"time_id__{second}",
        )
        df_feature = df_feature.drop([f"time_id__{second}"], axis=1)

    df_feature = df_feature.add_prefix("trade_")
    stock_id = file_path.split("=")[1]
    df_feature["row_id"] = df_feature["trade_time_id_"].apply(
        lambda x: f"{stock_id}-{x}"
    )
    df_feature = df_feature.drop(["trade_time_id_"], axis=1)

    return df_feature


def preprocessor(list_stock_ids: List[int], is_train: bool = True) -> pd.DataFrame:
    from joblib import Parallel, delayed  # parallel computing to save time

    df = pd.DataFrame()

    def for_joblib(stock_id: int) -> pd.DataFrame:
        if is_train:
            file_path_book = data_dir + "book_train.parquet/stock_id=" + str(stock_id)
            file_path_trade = data_dir + "trade_train.parquet/stock_id=" + str(stock_id)
        else:
            file_path_book = data_dir + "book_test.parquet/stock_id=" + str(stock_id)
            file_path_trade = data_dir + "trade_test.parquet/stock_id=" + str(stock_id)

        df_tmp = pd.merge(
            preprocessor_book(file_path_book),
            preprocessor_trade(file_path_trade),
            on="row_id",
            how="left",
        )

        return pd.concat([df, df_tmp])

    df = Parallel(n_jobs=-1, verbose=1)(
        delayed(for_joblib)(stock_id) for stock_id in list_stock_ids
    )

    df = pd.concat(df, ignore_index=True)
    return df


def load_dataset(path: str) -> Tuple:
    train = pd.read_csv(path + "train.csv")
    train_ids = train.stock_id.unique()
    df_train = preprocessor(list_stock_ids=train_ids, is_train=True)
    train["row_id"] = train["stock_id"].astype(str) + "-" + train["time_id"].astype(str)
    train = train[["row_id", "target"]]
    df_train = train.merge(df_train, on=["row_id"], how="left")

    test = pd.read_csv(path + "test.csv")
    test_ids = test.stock_id.unique()
    df_test = preprocessor(list_stock_ids=test_ids, is_train=False)
    df_test = test.merge(df_test, on=["row_id"], how="left")

    df_train["stock_id"] = df_train["stock_id"].astype(int)
    df_test["stock_id"] = df_test["stock_id"].astype(int)

    return df_train, df_test
