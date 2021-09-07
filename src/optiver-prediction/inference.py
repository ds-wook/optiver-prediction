import pandas as pd
from data.dataset import add_tau_feature, create_agg_features, load_test


def _main():
    path = "../../input/optiver-realized-volatility-prediction/"
    train = pd.read_pickle(path + "cluster_train.pkl")
    test = load_test(path)

    train, test = add_tau_feature(train, test)
    train, test = create_agg_features(train, test, path)


if __name__ == "__main__":
    _main()
