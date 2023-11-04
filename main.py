import torch
import wandb
import yaml

# start a new wandb run to track this script


def main():
    # project total config
    with open("model_config.yaml","r") as file:
        cfg = yaml.safe_load(file)

    # taining visualization setting
    wandb.init(
        # set the wandb project where this run will be logged
        project="RTDETR_Refactor",
        name = "test",
        # track hyperparameters and run metadata
        config=cfg
    )
    pass


if __name__ == '__main__':
    main()
