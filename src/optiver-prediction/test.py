# data directory
data_dir = "../input/optiver-realized-volatility-prediction/"

# Function to calculate first WAP
def calc_wap1(df):
    wap = (df["bid_price1"] * df["ask_size1"] + df["ask_price1"] * df["bid_size1"]) / (
        df["bid_size1"] + df["ask_size1"]
    )
    return wap


# Function to calculate second WAP
def calc_wap2(df):
    wap = (df["bid_price2"] * df["ask_size2"] + df["ask_price2"] * df["bid_size2"]) / (
        df["bid_size2"] + df["ask_size2"]
    )
    return wap


def calc_wap3(df):
    wap = (df["bid_price1"] * df["bid_size1"] + df["ask_price1"] * df["ask_size1"]) / (
        df["bid_size1"] + df["ask_size1"]
    )
    return wap


def calc_wap4(df):
    wap = (df["bid_price2"] * df["bid_size2"] + df["ask_price2"] * df["ask_size2"]) / (
        df["bid_size2"] + df["ask_size2"]
    )
    return wap


# Function to calculate the log of the return
# Remember that logb(x / y) = logb(x) - logb(y)
def log_return(series):
    return np.log(series).diff()


# Calculate the realized volatility
def realized_volatility(series):
    return np.sqrt(np.sum(series ** 2))


# Function to count unique elements of a series
def count_unique(series):
    return len(np.unique(series))


