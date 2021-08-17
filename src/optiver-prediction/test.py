import warnings
from pathlib import Path

import hydra
import pandas as pd
from omegaconf import DictConfig, OmegaConf

warnings.filterwarnings("ignore")


@hydra.main(config_path="../../config/train/", config_name="lgbm_train.yml")
def _main(cfg: DictConfig):
    config = OmegaConf.to_yaml(cfg)
    print(config)


if __name__ == "__main__":
    _main()
