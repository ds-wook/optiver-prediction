from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from hydra.utils import to_absolute_path
from joblib import Parallel, delayed
from sklearn.cluster import KMeans
from tqdm import tqdm

data_dir = to_absolute_path("../../input/optiver-realized-volatility-prediction/") + "/"


# Function to calculate first WAP
def calc_wap1(df: pd.DataFrame) -> pd.Series:
    wap = (df["bid_price1"] * df["ask_size1"] + df["ask_price1"] * df["bid_size1"]) / (
        df["bid_size1"] + df["ask_size1"]
    )
    return wap


# Function to calculate second WAP
def calc_wap2(df: pd.DataFrame) -> pd.Series:
    wap = (df["bid_price2"] * df["ask_size2"] + df["ask_price2"] * df["bid_size2"]) / (
        df["bid_size2"] + df["ask_size2"]
    )
    return wap


def calc_wap3(df: pd.DataFrame) -> pd.Series:
    wap = (df["bid_price1"] * df["bid_size1"] + df["ask_price1"] * df["ask_size1"]) / (
        df["bid_size1"] + df["ask_size1"]
    )
    return wap


def calc_wap4(df: pd.DataFrame) -> pd.Series:
    wap = (df["bid_price2"] * df["bid_size2"] + df["ask_price2"] * df["ask_size2"]) / (
        df["bid_size2"] + df["ask_size2"]
    )
    return wap


def encode_mean(column: str, df: pd.DataFrame) -> float:
    avg = df.groupby("time_id")[column].transform("mean")
    return np.abs(df[column].sub(avg).div(avg))


# Function to calculate the log of the return
# Remember that logb(x / y) = logb(x) - logb(y)
def log_return(series: pd.DataFrame) -> np.ndarray:
    return np.log(series).diff()


# Calculate the realized volatility
def realized_volatility(series: pd.DataFrame) -> np.ndarray:
    return np.sqrt(np.sum(series ** 2))


# Function to count unique elements of a series
def count_unique(series: pd.DataFrame) -> np.ndarray:
    return len(np.unique(series))


def realized_quarticity(series: pd.DataFrame) -> np.ndarray:
    return np.sum(series ** 4) * series.shape[0] / 3


def realized_quadpower_quarticity(series: pd.DataFrame) -> np.ndarray:
    series = series.rolling(window=4).apply(np.product, raw=True)
    return (np.sum(series) * series.shape[0] * (np.pi ** 2)) / 4


def realized_1(series: pd.DataFrame) -> np.ndarray:
    return np.sqrt(np.sum(series ** 4) / (6 * np.sum(series ** 2)))


def realized_2(series: pd.DataFrame) -> np.ndarray:
    return np.sqrt(
        ((np.pi ** 2) * np.sum(series.rolling(window=4).apply(np.product, raw=True)))
        / (8 * np.sum(series ** 2))
    )


# Function to read our base train and test set
def read_train_test(path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(path + "train.csv")
    test = pd.read_csv(path + "test.csv")
    # Create a key to merge with book and trade data
    train["row_id"] = train["stock_id"].astype(str) + "-" + train["time_id"].astype(str)
    test["row_id"] = test["stock_id"].astype(str) + "-" + test["time_id"].astype(str)
    print(f"Our training set has {train.shape[0]} rows")
    return train, test


# Function to read our base train and test set
def read_test(path: str) -> pd.DataFrame:
    test = pd.read_csv(path + "test.csv")
    # Create a key to merge with book and trade data
    test["row_id"] = test["stock_id"].astype(str) + "-" + test["time_id"].astype(str)
    return test


# Function to preprocess book data (for each stock id)
def book_preprocessor(file_path: str):
    df = pd.read_parquet(file_path)
    # Calculate Wap
    df["wap1"] = calc_wap1(df)
    df["wap2"] = calc_wap2(df)
    df["wap3"] = calc_wap3(df)
    df["wap4"] = calc_wap4(df)
    # Calculate log returns
    df["log_return1"] = df.groupby(["time_id"])["wap1"].apply(log_return)
    df["log_return2"] = df.groupby(["time_id"])["wap2"].apply(log_return)
    df["log_return3"] = df.groupby(["time_id"])["wap3"].apply(log_return)
    df["log_return4"] = df.groupby(["time_id"])["wap4"].apply(log_return)
    # Calculate wap balance
    df["wap_balance"] = abs(df["wap1"] - df["wap2"])
    # Calculate spread
    df["price_spread"] = (df["ask_price1"] - df["bid_price1"]) / (
        (df["ask_price1"] + df["bid_price1"]) / 2
    )
    df["price_spread2"] = (df["ask_price2"] - df["bid_price2"]) / (
        (df["ask_price2"] + df["bid_price2"]) / 2
    )
    df["bid_spread"] = df["bid_price1"] - df["bid_price2"]
    df["ask_spread"] = df["ask_price1"] - df["ask_price2"]
    df["bid_ask_spread"] = abs(df["bid_spread"] - df["ask_spread"])
    df["total_volume"] = (df["ask_size1"] + df["ask_size2"]) + (
        df["bid_size1"] + df["bid_size2"]
    )
    df["volume_imbalance"] = abs(
        (df["ask_size1"] + df["ask_size2"]) - (df["bid_size1"] + df["bid_size2"])
    )

    # Dict for aggregations
    create_feature_dict = {
        "wap1": [np.sum, np.std],
        "wap2": [np.sum, np.std],
        "wap3": [np.sum, np.std],
        "wap4": [np.sum, np.std],
        "log_return1": [realized_volatility],
        "log_return2": [realized_volatility],
        "log_return3": [realized_volatility],
        "log_return4": [realized_volatility],
        "wap_balance": [np.sum, np.max],
        "price_spread": [np.sum, np.max],
        "price_spread2": [np.sum, np.max],
        "bid_spread": [np.sum, np.max],
        "ask_spread": [np.sum, np.max],
        "total_volume": [np.sum, np.max],
        "volume_imbalance": [np.sum, np.max],
        "bid_ask_spread": [np.sum, np.max],
    }
    create_feature_dict_time = {
        "log_return1": [realized_volatility],
        "log_return2": [realized_volatility],
        "log_return3": [realized_volatility],
        "log_return4": [realized_volatility],
    }

    # Function to get group stats for different windows (seconds in bucket)
    def get_stats_window(
        fe_dict: Dict[str, List[Any]], seconds_in_bucket: int, add_suffix: bool = False
    ) -> pd.DataFrame:
        # Group by the window
        df_feature = (
            df[df["seconds_in_bucket"] >= seconds_in_bucket]
            .groupby(["time_id"])
            .agg(fe_dict)
            .reset_index()
        )
        # Rename columns joining suffix
        df_feature.columns = ["_".join(col) for col in df_feature.columns]
        # Add a suffix to differentiate windows
        if add_suffix:
            df_feature = df_feature.add_suffix("_" + str(seconds_in_bucket))
        return df_feature

    # Get the stats for different windows
    df_feature = get_stats_window(
        create_feature_dict, seconds_in_bucket=0, add_suffix=False
    )
    df_feature_500 = get_stats_window(
        create_feature_dict_time, seconds_in_bucket=500, add_suffix=True
    )
    df_feature_400 = get_stats_window(
        create_feature_dict_time, seconds_in_bucket=400, add_suffix=True
    )
    df_feature_300 = get_stats_window(
        create_feature_dict_time, seconds_in_bucket=300, add_suffix=True
    )
    df_feature_200 = get_stats_window(
        create_feature_dict_time, seconds_in_bucket=200, add_suffix=True
    )
    df_feature_100 = get_stats_window(
        create_feature_dict_time, seconds_in_bucket=100, add_suffix=True
    )

    # Merge all
    df_feature = df_feature.merge(
        df_feature_500, how="left", left_on="time_id_", right_on="time_id__500"
    )
    df_feature = df_feature.merge(
        df_feature_400, how="left", left_on="time_id_", right_on="time_id__400"
    )
    df_feature = df_feature.merge(
        df_feature_300, how="left", left_on="time_id_", right_on="time_id__300"
    )
    df_feature = df_feature.merge(
        df_feature_200, how="left", left_on="time_id_", right_on="time_id__200"
    )
    df_feature = df_feature.merge(
        df_feature_100, how="left", left_on="time_id_", right_on="time_id__100"
    )
    # Drop unnecesary time_ids
    df_feature.drop(
        [
            "time_id__500",
            "time_id__400",
            "time_id__300",
            "time_id__200",
            "time_id__100",
        ],
        axis=1,
        inplace=True,
    )

    # Create row_id so we can merge
    stock_id = file_path.split("=")[1]
    df_feature["row_id"] = df_feature["time_id_"].apply(lambda x: f"{stock_id}-{x}")
    df_feature.drop(["time_id_"], axis=1, inplace=True)
    return df_feature


# Function to preprocess trade data (for each stock id)
def trade_preprocessor(file_path: str) -> pd.DataFrame:
    df = pd.read_parquet(file_path)
    df["log_return"] = df.groupby("time_id")["price"].apply(log_return)
    df["amount"] = df["price"] * df["size"]
    # Dict for aggregations
    create_feature_dict = {
        "log_return": [realized_volatility],
        "seconds_in_bucket": [count_unique],
        "size": [np.sum, np.max, np.min],
        "order_count": [np.sum, np.max],
        "amount": [np.sum, np.max, np.min],
    }
    create_feature_dict_time = {
        "log_return": [realized_volatility],
        "seconds_in_bucket": [count_unique],
        "size": [np.sum],
        "order_count": [np.sum],
    }

    # Function to get group stats for different windows (seconds in bucket)
    def get_stats_window(fe_dict, seconds_in_bucket, add_suffix=False):
        # Group by the window
        df_feature = (
            df[df["seconds_in_bucket"] >= seconds_in_bucket]
            .groupby(["time_id"])
            .agg(fe_dict)
            .reset_index()
        )
        # Rename columns joining suffix
        df_feature.columns = ["_".join(col) for col in df_feature.columns]
        # Add a suffix to differentiate windows
        if add_suffix:
            df_feature = df_feature.add_suffix("_" + str(seconds_in_bucket))
        return df_feature

    # Get the stats for different windows
    df_feature = get_stats_window(
        create_feature_dict, seconds_in_bucket=0, add_suffix=False
    )
    df_feature_500 = get_stats_window(
        create_feature_dict_time, seconds_in_bucket=500, add_suffix=True
    )
    df_feature_400 = get_stats_window(
        create_feature_dict_time, seconds_in_bucket=400, add_suffix=True
    )
    df_feature_300 = get_stats_window(
        create_feature_dict_time, seconds_in_bucket=300, add_suffix=True
    )
    df_feature_200 = get_stats_window(
        create_feature_dict_time, seconds_in_bucket=200, add_suffix=True
    )
    df_feature_100 = get_stats_window(
        create_feature_dict_time, seconds_in_bucket=100, add_suffix=True
    )

    def tendency(price: np.ndarray, vol: np.ndarray) -> float:
        df_diff = np.diff(price)
        val = (df_diff / price[1:]) * 100
        power = np.sum(val * vol[1:])
        return power

    lis = []
    for n_time_id in df["time_id"].unique():
        df_id = df[df["time_id"] == n_time_id]
        tendencyV = tendency(df_id["price"].values, df_id["size"].values)
        f_max = np.sum(df_id["price"].values > np.mean(df_id["price"].values))
        f_min = np.sum(df_id["price"].values < np.mean(df_id["price"].values))
        df_max = np.sum(np.diff(df_id["price"].values) > 0)
        df_min = np.sum(np.diff(df_id["price"].values) < 0)
        # new
        abs_diff = np.median(
            np.abs(df_id["price"].values - np.mean(df_id["price"].values))
        )
        energy = np.mean(df_id["price"].values ** 2)
        iqr_p = np.percentile(df_id["price"].values, 75) - np.percentile(
            df_id["price"].values, 25
        )

        # vol vars
        abs_diff_v = np.median(
            np.abs(df_id["size"].values - np.mean(df_id["size"].values))
        )
        energy_v = np.sum(df_id["size"].values ** 2)
        iqr_p_v = np.percentile(df_id["size"].values, 75) - np.percentile(
            df_id["size"].values, 25
        )

        lis.append(
            {
                "time_id": n_time_id,
                "tendency": tendencyV,
                "f_max": f_max,
                "f_min": f_min,
                "df_max": df_max,
                "df_min": df_min,
                "abs_diff": abs_diff,
                "energy": energy,
                "iqr_p": iqr_p,
                "abs_diff_v": abs_diff_v,
                "energy_v": energy_v,
                "iqr_p_v": iqr_p_v,
            }
        )

    df_lr = pd.DataFrame(lis)

    df_feature = df_feature.merge(
        df_lr, how="left", left_on="time_id_", right_on="time_id"
    )

    # Merge all
    df_feature = df_feature.merge(
        df_feature_500, how="left", left_on="time_id_", right_on="time_id__500"
    )
    df_feature = df_feature.merge(
        df_feature_400, how="left", left_on="time_id_", right_on="time_id__400"
    )
    df_feature = df_feature.merge(
        df_feature_300, how="left", left_on="time_id_", right_on="time_id__300"
    )
    df_feature = df_feature.merge(
        df_feature_200, how="left", left_on="time_id_", right_on="time_id__200"
    )
    df_feature = df_feature.merge(
        df_feature_100, how="left", left_on="time_id_", right_on="time_id__100"
    )
    # Drop unnecesary time_ids
    df_feature.drop(
        [
            "time_id__500",
            "time_id__400",
            "time_id__300",
            "time_id__200",
            "time_id",
            "time_id__100",
        ],
        axis=1,
        inplace=True,
    )

    df_feature = df_feature.add_prefix("trade_")
    stock_id = file_path.split("=")[1]
    df_feature["row_id"] = df_feature["trade_time_id_"].apply(
        lambda x: f"{stock_id}-{x}"
    )
    df_feature.drop(["trade_time_id_"], axis=1, inplace=True)
    return df_feature


def encode_timeid(
    train: pd.DataFrame, test: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    columns_to_encode = [
        "wap1_sum",
        "wap2_sum",
        "wap3_sum",
        "wap4_sum",
        "log_return1_realized_volatility",
        "log_return2_realized_volatility",
        "log_return3_realized_volatility",
        "log_return4_realized_volatility",
        "wap_balance_sum",
        "price_spread_sum",
        "price_spread2_sum",
        "bid_spread_sum",
        "ask_spread_sum",
        "total_volume_sum",
        "volume_imbalance_sum",
        "bid_ask_spread_sum",
        "trade_log_return_realized_volatility",
        "trade_seconds_in_bucket_count_unique",
        "trade_size_sum",
        "trade_order_count_sum",
        "trade_amount_sum",
        "trade_tendency",
        "trade_f_max",
        "trade_df_max",
        "trade_abs_diff",
        "trade_energy",
        "trade_iqr_p",
        "trade_abs_diff_v",
        "trade_energy_v",
        "trade_iqr_p_v",
    ]

    df_aux = Parallel(n_jobs=-1, verbose=1)(
        delayed(encode_mean)(column, train) for column in columns_to_encode
    )
    # Get group stats of time_id and stock_id
    train = pd.concat(
        [train] + [x.rename(x.name + "_timeid_encoded") for x in df_aux], axis=1
    )
    del df_aux

    df_aux = Parallel(n_jobs=-1, verbose=1)(
        delayed(encode_mean)(column, test) for column in columns_to_encode
    )
    # Get group stats of time_id and stock_id
    test = pd.concat(
        [test] + [x.rename(x.name + "_timeid_encoded") for x in df_aux], axis=1
    )
    del df_aux
    return train, test


# Function to get group stats for the stock_id and time_id
def get_time_stock(df: pd.DataFrame) -> pd.DataFrame:
    vol_cols = [
        "log_return1_realized_volatility",
        "log_return2_realized_volatility",
        "log_return1_realized_volatility_400",
        "log_return2_realized_volatility_400",
        "log_return1_realized_volatility_300",
        "log_return2_realized_volatility_300",
        "log_return1_realized_volatility_200",
        "log_return2_realized_volatility_200",
        "trade_log_return_realized_volatility",
        "trade_log_return_realized_volatility_400",
        "trade_log_return_realized_volatility_300",
        "trade_log_return_realized_volatility_200",
    ]

    # Group by the stock id
    df_stock_id = (
        df.groupby(["stock_id"])[vol_cols]
        .agg(
            [
                "mean",
                "std",
                "max",
                "min",
            ]
        )
        .reset_index()
    )
    # Rename columns joining suffix
    df_stock_id.columns = ["_".join(col) for col in df_stock_id.columns]
    df_stock_id = df_stock_id.add_suffix("_" + "stock")

    # Group by the stock id
    df_time_id = (
        df.groupby(["time_id"])[vol_cols]
        .agg(
            [
                "mean",
                "std",
                "max",
                "min",
            ]
        )
        .reset_index()
    )
    # Rename columns joining suffix
    df_time_id.columns = ["_".join(col) for col in df_time_id.columns]
    df_time_id = df_time_id.add_suffix("_" + "time")

    # Merge with original dataframe
    df = df.merge(
        df_stock_id, how="left", left_on=["stock_id"], right_on=["stock_id__stock"]
    )
    df = df.merge(
        df_time_id, how="left", left_on=["time_id"], right_on=["time_id__time"]
    )
    df.drop(["stock_id__stock", "time_id__time"], axis=1, inplace=True)
    return df


# Funtion to make preprocessing function in parallel (for each stock id)
def preprocessor(list_stock_ids: np.ndarray, is_train: bool = True) -> pd.DataFrame:

    # Parrallel for loop
    def for_joblib(stock_id: int) -> pd.DataFrame:
        # Train
        if is_train:
            file_path_book = data_dir + "book_train.parquet/stock_id=" + str(stock_id)
            file_path_trade = data_dir + "trade_train.parquet/stock_id=" + str(stock_id)
        # Test
        else:
            file_path_book = data_dir + "book_test.parquet/stock_id=" + str(stock_id)
            file_path_trade = data_dir + "trade_test.parquet/stock_id=" + str(stock_id)

        # Preprocess book and trade data and merge them
        df_tmp = pd.merge(
            book_preprocessor(file_path_book),
            trade_preprocessor(file_path_trade),
            on="row_id",
            how="left",
        )

        # Return the merge dataframe
        return df_tmp

    # Use parallel api to call paralle for loop
    df = Parallel(n_jobs=-1, verbose=1)(
        delayed(for_joblib)(stock_id) for stock_id in list_stock_ids
    )
    # Concatenate all the dataframes that return from Parallel
    df = pd.concat(df, ignore_index=True)
    return df


# replace by order sum (tau)
def add_tau_feature(
    train: pd.DataFrame, test: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train["size_tau"] = np.sqrt(1 / train["trade_seconds_in_bucket_count_unique"])
    test["size_tau"] = np.sqrt(1 / test["trade_seconds_in_bucket_count_unique"])
    # train['size_tau_450'] = np.sqrt( 1/ train['trade_seconds_in_bucket_count_unique_450'] )
    # test['size_tau_450'] = np.sqrt( 1/ test['trade_seconds_in_bucket_count_unique_450'] )
    train["size_tau_400"] = np.sqrt(
        1 / train["trade_seconds_in_bucket_count_unique_400"]
    )
    test["size_tau_400"] = np.sqrt(1 / test["trade_seconds_in_bucket_count_unique_400"])
    train["size_tau_300"] = np.sqrt(
        1 / train["trade_seconds_in_bucket_count_unique_300"]
    )
    test["size_tau_300"] = np.sqrt(1 / test["trade_seconds_in_bucket_count_unique_300"])
    # train['size_tau_150'] = np.sqrt( 1/ train['trade_seconds_in_bucket_count_unique_150'] )
    # test['size_tau_150'] = np.sqrt( 1/ test['trade_seconds_in_bucket_count_unique_150'] )
    train["size_tau_200"] = np.sqrt(
        1 / train["trade_seconds_in_bucket_count_unique_200"]
    )
    test["size_tau_200"] = np.sqrt(1 / test["trade_seconds_in_bucket_count_unique_200"])
    train["size_tau2"] = np.sqrt(1 / train["trade_order_count_sum"])
    test["size_tau2"] = np.sqrt(1 / test["trade_order_count_sum"])
    # train['size_tau2_450'] = np.sqrt( 0.25/ train['trade_order_count_sum'] )
    # test['size_tau2_450'] = np.sqrt( 0.25/ test['trade_order_count_sum'] )
    train["size_tau2_400"] = np.sqrt(0.33 / train["trade_order_count_sum"])
    test["size_tau2_400"] = np.sqrt(0.33 / test["trade_order_count_sum"])
    train["size_tau2_300"] = np.sqrt(0.5 / train["trade_order_count_sum"])
    test["size_tau2_300"] = np.sqrt(0.5 / test["trade_order_count_sum"])
    # train['size_tau2_150'] = np.sqrt( 0.75/ train['trade_order_count_sum'] )
    # test['size_tau2_150'] = np.sqrt( 0.75/ test['trade_order_count_sum'] )
    train["size_tau2_200"] = np.sqrt(0.66 / train["trade_order_count_sum"])
    test["size_tau2_200"] = np.sqrt(0.66 / test["trade_order_count_sum"])

    # delta tau
    train["size_tau2_d"] = train["size_tau2_400"] - train["size_tau2"]
    test["size_tau2_d"] = test["size_tau2_400"] - test["size_tau2"]

    return train, test


def create_agg_features(
    train: pd.DataFrame, test: pd.DataFrame, path: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Making agg features
    train_p = pd.read_csv(path + "train.csv")
    train_p = train_p.pivot(index="time_id", columns="stock_id", values="target")
    corr = train_p.corr()
    ids = corr.index
    kmeans = KMeans(n_clusters=7, random_state=0).fit(corr.values)
    indexes = [
        [(x - 1) for x in ((ids + 1) * (kmeans.labels_ == n)) if x > 0]
        for n in tqdm(range(7))
    ]

    mat = []
    mat_test = []
    n = 0
    for ind in tqdm(indexes):
        new_df = train.loc[train["stock_id"].isin(ind)]
        new_df = new_df.groupby(["time_id"]).agg(np.nanmean)
        new_df.loc[:, "stock_id"] = str(n) + "c1"
        mat.append(new_df)
        new_df = test.loc[test["stock_id"].isin(ind)]
        new_df = new_df.groupby(["time_id"]).agg(np.nanmean)
        new_df.loc[:, "stock_id"] = str(n) + "c1"
        mat_test.append(new_df)
        n += 1

    mat1 = pd.concat(mat).reset_index()
    mat1.drop(columns=["target"], inplace=True)
    mat2 = pd.concat(mat_test).reset_index()

    mat2 = pd.concat([mat2, mat1.loc[mat1.time_id == 5]])

    mat1 = mat1.pivot(index="time_id", columns="stock_id")
    mat1.columns = ["_".join(x) for x in tqdm(mat1.columns.tolist())]
    mat1.reset_index(inplace=True)

    mat2 = mat2.pivot(index="time_id", columns="stock_id")
    mat2.columns = ["_".join(x) for x in tqdm(mat2.columns.tolist())]
    mat2.reset_index(inplace=True)

    prefix = [
        "log_return1_realized_volatility",
        "total_volume_sum",
        "trade_size_sum",
        "trade_order_count_sum",
        "price_spread_sum",
        "bid_spread_sum",
        "ask_spread_sum",
        "volume_imbalance_sum",
        "bid_ask_spread_sum",
        "size_tau2",
    ]
    selected_cols = mat1.filter(
        regex="|".join(f"^{x}.(0|1|3|4|6)c1" for x in tqdm(prefix))
    ).columns.tolist()
    selected_cols.append("time_id")
    train_m = pd.merge(train, mat1[selected_cols], how="left", on="time_id")
    test_m = pd.merge(test, mat2[selected_cols], how="left", on="time_id")

    # filling missing values with train means
    features = [
        col
        for col in train_m.columns.tolist()
        if col not in ["time_id", "target", "row_id"]
    ]
    train_m[features] = train_m[features].fillna(train_m[features].mean())
    test_m[features] = test_m[features].fillna(train_m[features].mean())

    return train_m, test_m


def network_agg_features(
    train: pd.DataFrame, test: pd.DataFrame, path: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Making agg features
    train_p = pd.read_csv(path + "train.csv")
    train_p = train_p.pivot(index="time_id", columns="stock_id", values="target")
    corr = train_p.corr()
    ids = corr.index
    kmeans = KMeans(n_clusters=7, random_state=0).fit(corr.values)
    indexes = [
        [(x - 1) for x in ((ids + 1) * (kmeans.labels_ == n)) if x > 0]
        for n in tqdm(range(7))
    ]

    mat = []
    mat_test = []
    n = 0
    for ind in tqdm(indexes):
        new_df = train.loc[train["stock_id"].isin(ind)]
        new_df = new_df.groupby(["time_id"]).agg(np.nanmean)
        new_df.loc[:, "stock_id"] = str(n) + "c1"
        mat.append(new_df)
        new_df = test.loc[test["stock_id"].isin(ind)]
        new_df = new_df.groupby(["time_id"]).agg(np.nanmean)
        new_df.loc[:, "stock_id"] = str(n) + "c1"
        mat_test.append(new_df)
        n += 1

    mat1 = pd.concat(mat).reset_index()
    mat1.drop(columns=["target"], inplace=True)
    mat2 = pd.concat(mat_test).reset_index()

    mat2 = pd.concat([mat2, mat1.loc[mat1.time_id == 5]])

    mat1 = mat1.pivot(index="time_id", columns="stock_id")
    mat1.columns = ["_".join(x) for x in tqdm(mat1.columns.tolist())]
    mat1.reset_index(inplace=True)

    mat2 = mat2.pivot(index="time_id", columns="stock_id")
    mat2.columns = ["_".join(x) for x in tqdm(mat2.columns.tolist())]
    mat2.reset_index(inplace=True)

    prefix = [
        "log_return1_realized_volatility",
        "total_volume_mean",
        "trade_size_mean",
        "trade_order_count_mean",
        "price_spread_mean",
        "bid_spread_mean",
        "ask_spread_mean",
        "volume_imbalance_mean",
        "bid_ask_spread_mean",
        "size_tau2",
    ]
    selected_cols = mat1.filter(
        regex="|".join(f"^{x}.(0|1|3|4|6)c1" for x in prefix)
    ).columns.tolist()
    selected_cols.append("time_id")

    train_m = pd.merge(train, mat1[selected_cols], how="left", on="time_id")
    test_m = pd.merge(test, mat2[selected_cols], how="left", on="time_id")

    # filling missing values with train means
    features = [
        col
        for col in train_m.columns.tolist()
        if col not in ["time_id", "target", "row_id"]
    ]
    train_m[features] = train_m[features].fillna(train_m[features].mean())
    test_m[features] = test_m[features].fillna(train_m[features].mean())

    return train_m, test_m


# Function to read our base train and test set
def load_test(path: str) -> pd.DataFrame:
    test = pd.read_csv(path + "test.csv")
    # Create a key to merge with book and trade data
    test["row_id"] = test["stock_id"].astype(str) + "-" + test["time_id"].astype(str)
    # Get unique stock ids
    test_stock_ids = test["stock_id"].unique()
    # Preprocess them using Parallel and our single stock id functions
    test_ = preprocessor(test_stock_ids, is_train=False)
    test = test.merge(test_, on=["row_id"], how="left")
    test = get_time_stock(test)
    return test


def load_dataset(path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Read train and test
    train, test = read_train_test(path)

    # Get unique stock ids
    train_stock_ids = train["stock_id"].unique()
    # Preprocess them using Parallel and our single stock id functions
    train_ = preprocessor(train_stock_ids, is_train=True)
    train = train.merge(train_, on=["row_id"], how="left")

    # Get unique stock ids
    test_stock_ids = test["stock_id"].unique()
    # Preprocess them using Parallel and our single stock id functions
    test_ = preprocessor(test_stock_ids, is_train=False)
    test = test.merge(test_, on=["row_id"], how="left")

    # Get group stats of time_id and stock_id
    train = get_time_stock(train)
    test = get_time_stock(test)

    print(f"Before Train Features: {train.shape}")
    print(f"Before Test Features: {test.shape}")
    train, test = add_tau_feature(train, test)
    print(f"Before Train Features: {train.shape}")
    print(f"Before Test Features: {test.shape}")
    train, test = create_agg_features(train, test, path)

    print(f"After Train Features: {train.shape}")
    print(f"After Test Features: {test.shape}")
    return train, test
