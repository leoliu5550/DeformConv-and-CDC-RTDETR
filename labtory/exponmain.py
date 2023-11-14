import yaml
import wandb 
import torch
import torch.nn as nn
with open("model_config.yaml") as file:
    cfg= yaml.safe_load(file)



wandb.init(
    # set the wandb project where this run will be logged
    project="RTDETR_Refactor",
    name = cfg['wandbname'],
    # track hyperparameters and run metadata
    config=cfg
)

model = nn.Sequential(
    nn.Linear(5,1)
)
loss_func = nn.MSELoss()

optimizer = torch.optim.SGD( model.parameters(),lr=0.1, momentum=0.9)

for epoch in range(100):
    x = torch.randn([5,5])
    output = model(x)
    loss = loss_func(x,output)
    loss_dict = {"NNloss":torch.randn([1]),
                "Loss":loss}
    print(loss_dict)
    
    # wandb.log()
    loss.backward()
    optimizer.step()

    for loss_name,loss_value in loss_dict.items():
        wandb.log({f"train-{loss_name}":loss_value})


    # wandb.log({epoch})