import wandb
import json

path = r"output/rtdetr_r50vd_6x_coco_deformConvRelu/log.txt"

cfg = {
    "epoch": 71, "n_parameters": 47239437
}
wandb.init(
    # set the wandb project where this run will be logged
    project="RTDETR_Refactor",
    name = "test_run",
    # # track hyperparameters and run metadata
    config=cfg
)
data = []
with open(path) as f:
    for line in f:
        data.append(json.loads(line))
        
def wandb_prefixlogs(loss_dict,train=True):
    if train:
        prefix = "train"
    else:
        prefix = "valid"
    
        
    for key,value in loss_dict.items():
        if key == "test_coco_eval_bbox":
            continue
        else:
            logs = {f"{prefix}.{key}":value}
            wandb.log(logs)

for rows in data:
    wandb_prefixlogs(rows)
