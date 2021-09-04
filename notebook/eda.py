# %%
import pandas as pd

train = pd.read_pickle("../input/optiver-realized-volatility-prediction/fea_train.pkl")
train.shape

# %%
train.columns.tolist()
# %%
