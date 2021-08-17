# %%
import hydra
import pandas as pd
from omegaconf import DictConfig


@hydra.main(config_path="../config/train/", config_name="lgbm_train.yml")
def _main(cfg: DictConfig):
    train = pd.read_csv(cfg.dataset.path + "train.csv")
    print(train)


if __name__ == "__main__":
    _main()

# %%
