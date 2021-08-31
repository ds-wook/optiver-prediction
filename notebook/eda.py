# %%
import pandas as pd

path = "../input/optiver-realized-volatility-prediction/"
train = pd.read_pickle(path + "fea0_train_best.pkl")
train.head()
# %%
train = pd.read_pickle(path + "fea2_train_best.pkl")
train.head()
# %%
print(train.columns.tolist())
# %%
