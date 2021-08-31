import hydra
from omegaconf import DictConfig


@hydra.main(config_path="../../config/train/", config_name="train.yaml")
def _main(cfg: DictConfig):
    # Hyperparammeters (optimized)
    params = dict(cfg.params)
    print(type(cfg.params))
    print(params)
    print(type(params))


if __name__ == "__main__":
    _main()
