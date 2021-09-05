# %%
import pandas as pd

train = pd.read_pickle("../input/optiver-realized-volatility-prediction/cluster_train.pkl")
print([c for c in train.columns if "power" in c])
# %%
train.columns.tolist()
# %%
