import wandb
import json

path = r"output/rtdetr_r50vd_6x_coco+deformConvv3/log.txt"

cfg = {
    "epoch": 71, "n_parameters": 50635843
}
wandb.init(
    # set the wandb project where this run will be logged
    project="RTDETR_Refactor",
    name = "RTDETR_deformConvS3S4S5",
    # # track hyperparameters and run metadata
    config=cfg
)
data = []
with open(path) as f:
    for line in f:
        data.append(json.loads(line))
        

for rows in data:
    # rows = rows.pop("test_coco_eval_bbox")
    del rows["test_coco_eval_bbox"]

    wandb.log(rows)