from pathlib import Path

import hydra
import pandas as pd
from omegaconf import DictConfig


@hydra.main(config_path="../../config/train/", config_name="lgbm_train.yaml")
def _main(cfg: DictConfig):
    path = hydra.utils.to_absolute_path(cfg.dataset.path)
    train = pd.read_csv(path + "/train.csv")
    print(train)


if __name__ == "__main__":
    _main()
