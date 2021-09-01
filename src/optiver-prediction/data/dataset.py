from typing import List, Tuple

import hydra
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.cluster import KMeans

data_dir = (
    hydra.utils.to_absolute_path("../../input/optiver-realized-volatility-prediction/")
    + "/"
)


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


# Function to calculate the log of the return
# Remember that logb(x / y) = logb(x) - logb(y)
def log_return(series: pd.Series) -> np.ndarray:
    return np.log(series).diff()


# Calculate the realized volatility
def realized_volatility(series: pd.Series) -> np.ndarray:
    return np.sqrt(np.sum(series ** 2))


# Function to count unique elements of a series
def count_unique(series: pd.Series) -> int:
    return len(np.unique(series))


# Function to read our base train and test set
def read_train_test(path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(path + "train.csv")
    test = pd.read_csv(path + "test.csv")
    # Create a key to merge with book and trade data
    train["row_id"] = train["stock_id"].astype(str) + "-" + train["time_id"].astype(str)
    test["row_id"] = test["stock_id"].astype(str) + "-" + test["time_id"].astype(str)
    print(f"Our training set has {train.shape[0]} rows")
    return train, test


# Function to preprocess book data (for each stock id)
def book_preprocessor(file_path: str) -> pd.DataFrame:
    df = pd.read_parquet(file_path)
    # Calculate Wap
    df["wap1"] = calc_wap1(df)
    df["wap2"] = calc_wap2(df)
    # Calculate log returns
    df["log_return1"] = df.groupby(["time_id"])["wap1"].apply(log_return)
    df["log_return2"] = df.groupby(["time_id"])["wap2"].apply(log_return)
    # Calculate wap balance
    df["wap_balance"] = abs(df["wap1"] - df["wap2"])
    # Calculate spread
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

    # Dict for aggregations
    create_feature_dict = {
        "wap1": [np.sum, np.mean, np.std],
        "wap2": [np.sum, np.mean, np.std],
        "log_return1": [np.sum, realized_volatility, np.mean, np.std],
        "log_return2": [np.sum, realized_volatility, np.mean, np.std],
        "wap_balance": [np.sum, np.mean, np.std],
        "price_spread": [np.sum, np.mean, np.std],
        "bid_spread": [np.sum, np.mean, np.std],
        "ask_spread": [np.sum, np.mean, np.std],
        "total_volume": [np.sum, np.mean, np.std],
        "volume_imbalance": [np.sum, np.mean, np.std],
    }

    # Function to get group stats for different windows (seconds in bucket)
    def get_stats_window(seconds_in_bucket, add_suffix=False):
        # Group by the window
        df_feature = (
            df[df["seconds_in_bucket"] >= seconds_in_bucket]
            .groupby(["time_id"])
            .agg(create_feature_dict)
            .reset_index()
        )
        # Rename columns joining suffix
        df_feature.columns = ["_".join(col) for col in df_feature.columns]
        # Add a suffix to differentiate windows
        if add_suffix:
            df_feature = df_feature.add_suffix("_" + str(seconds_in_bucket))
        return df_feature

    # Get the stats for different windows
    df_feature = get_stats_window(seconds_in_bucket=0, add_suffix=False)
    df_feature_500 = get_stats_window(seconds_in_bucket=500, add_suffix=True)
    df_feature_400 = get_stats_window(seconds_in_bucket=400, add_suffix=True)
    df_feature_300 = get_stats_window(seconds_in_bucket=300, add_suffix=True)
    df_feature_200 = get_stats_window(seconds_in_bucket=200, add_suffix=True)
    df_feature_100 = get_stats_window(seconds_in_bucket=100, add_suffix=True)

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

    # Dict for aggregations
    create_feature_dict = {
        "log_return": [realized_volatility],
        "seconds_in_bucket": [count_unique],
        "size": [np.sum],
        "order_count": [np.mean],
    }

    # Function to get group stats for different windows (seconds in bucket)
    def get_stats_window(seconds_in_bucket, add_suffix=False):
        # Group by the window
        df_feature = (
            df[df["seconds_in_bucket"] >= seconds_in_bucket]
            .groupby(["time_id"])
            .agg(create_feature_dict)
            .reset_index()
        )
        # Rename columns joining suffix
        df_feature.columns = ["_".join(col) for col in df_feature.columns]
        # Add a suffix to differentiate windows
        if add_suffix:
            df_feature = df_feature.add_suffix("_" + str(seconds_in_bucket))
        return df_feature

    # Get the stats for different windows
    df_feature = get_stats_window(seconds_in_bucket=0, add_suffix=False)
    df_feature_500 = get_stats_window(seconds_in_bucket=500, add_suffix=True)
    df_feature_400 = get_stats_window(seconds_in_bucket=400, add_suffix=True)
    df_feature_300 = get_stats_window(seconds_in_bucket=300, add_suffix=True)
    df_feature_200 = get_stats_window(seconds_in_bucket=200, add_suffix=True)
    df_feature_100 = get_stats_window(seconds_in_bucket=100, add_suffix=True)

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

    df_feature = df_feature.add_prefix("trade_")
    stock_id = file_path.split("=")[1]
    df_feature["row_id"] = df_feature["trade_time_id_"].apply(
        lambda x: f"{stock_id}-{x}"
    )
    df_feature.drop(["trade_time_id_"], axis=1, inplace=True)
    return df_feature


# Function to get group stats for the stock_id and time_id
def get_time_stock(df: pd.DataFrame) -> pd.DataFrame:
    # Get realized volatility columns
    #     vol_cols = ['log_return1_realized_volatility', 'log_return2_realized_volatility', 'log_return1_realized_volatility_450', 'log_return2_realized_volatility_450',
    #                 'log_return1_realized_volatility_300', 'log_return2_realized_volatility_300', 'log_return1_realized_volatility_150', 'log_return2_realized_volatility_150',
    #                 'trade_log_return_realized_volatility', 'trade_log_return_realized_volatility_450', 'trade_log_return_realized_volatility_300', 'trade_log_return_realized_volatility_150']
    vol_cols = [
        "log_return1_realized_volatility",
        "log_return2_realized_volatility",
        "log_return1_realized_volatility_500",
        "log_return2_realized_volatility_500",
        "log_return1_realized_volatility_400",
        "log_return2_realized_volatility_400",
        "log_return1_realized_volatility_300",
        "log_return2_realized_volatility_300",
        "log_return1_realized_volatility_200",
        "log_return2_realized_volatility_200",
        "log_return1_realized_volatility_100",
        "log_return2_realized_volatility_100",
        "trade_log_return_realized_volatility",
        "trade_log_return_realized_volatility_500",
        "trade_log_return_realized_volatility_400",
        "trade_log_return_realized_volatility_300",
        "trade_log_return_realized_volatility_200",
        "trade_log_return_realized_volatility_100",
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
def preprocessor(list_stock_ids: List[str], is_train: bool = True) -> pd.DataFrame:
    # Parrallel for loop
    def for_joblib(stock_id):
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


def create_agg_features(train, test):

    # Making agg features

    train_p = pd.read_csv("../input/optiver-realized-volatility-prediction/train.csv")
    train_p = train_p.pivot(index="time_id", columns="stock_id", values="target")
    corr = train_p.corr()
    ids = corr.index
    kmeans = KMeans(n_clusters=7, random_state=0).fit(corr.values)
    indexes = []
    for n in range(7):
        indexes.append([(x - 1) for x in ((ids + 1) * (kmeans.labels_ == n)) if x > 0])

    mat = []
    matTest = []
    n = 0
    for ind in indexes:
        new_df = train.loc[train["stock_id"].isin(ind)]
        new_df = new_df.groupby(["time_id"]).agg(np.nanmean)
        new_df.loc[:, "stock_id"] = str(n) + "c1"
        mat.append(new_df)
        new_df = test.loc[test["stock_id"].isin(ind)]
        new_df = new_df.groupby(["time_id"]).agg(np.nanmean)
        new_df.loc[:, "stock_id"] = str(n) + "c1"
        matTest.append(new_df)
        n += 1

    mat1 = pd.concat(mat).reset_index()
    mat1.drop(columns=["target"], inplace=True)
    mat2 = pd.concat(matTest).reset_index()

    mat2 = pd.concat([mat2, mat1.loc[mat1.time_id == 5]])

    mat1 = mat1.pivot(index="time_id", columns="stock_id")
    mat1.columns = ["_".join(x) for x in mat1.columns.ravel()]
    mat1.reset_index(inplace=True)

    mat2 = mat2.pivot(index="time_id", columns="stock_id")
    mat2.columns = ["_".join(x) for x in mat2.columns.ravel()]
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


def load_dataset(path: str) -> Tuple:
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

    return train, test
