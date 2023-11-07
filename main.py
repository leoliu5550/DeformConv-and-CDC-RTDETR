import torch
from torch.optim import Adam
from torch.utils.data import  DataLoader
import wandb

import yaml
from data import *
from model import *
from model import rtdetr


def main():
    cfg_path = "model_config.yaml"
    with open(cfg_path,"r") as file:
        cfg = yaml.safe_load(file)
    # start a new wandb run to track this script
    # wandb.init(
    #     # set the wandb project where this run will be logged
    #     project="RTDETR_Refactor",
    #     name = "testexp1",
    #     # track hyperparameters and run metadata
    #     config=cfg
    # )
    # project total config
    training_cfg = cfg['train']
    data_cfg = cfg['data']
    opt_cfg = cfg['optimizer']['adam']
    loss_cfg = cfg['LOSS']

    def collate_fn(batch):
        return tuple(zip(*batch))
        
    train_dataset = DataLoader(
        CocoDetection(
            img_folder = data_cfg['train_dataloader']['dataset']['img_folder'],
            ann_file = data_cfg['train_dataloader']['dataset']['ann_file'],
            return_masks = False),
        batch_size=data_cfg['train_dataloader']['batch_size'],
        shuffle=data_cfg['train_dataloader']['shuffle'],
        collate_fn=collate_fn
    )

    valid_dataset = DataLoader(
        CocoDetection(
            img_folder = data_cfg['val_dataloader']['dataset']['img_folder'],
            ann_file = data_cfg['val_dataloader']['dataset']['ann_file'],
            return_masks = False),
        batch_size=data_cfg['val_dataloader']['batch_size'],
        shuffle=data_cfg['val_dataloader']['shuffle'],
        collate_fn=collate_fn
    )

    model = rtdetr(cfg_path).to(cfg['device'])

    optimizer = Adam(model.parameters(),
                    lr = opt_cfg['lr'],
                    betas= tuple(opt_cfg['betas']),
                    eps = opt_cfg['eps'],
                    weight_decay = opt_cfg['weight_decay'],
                    amsgrad = opt_cfg['amsgrad']

    )
    Hungarian = HungarianMatcher(
        weight_dict = loss_cfg['matcher']['weight_dict'],
        use_focal_loss=False, 
        alpha=loss_cfg['matcher']['alpha'], 
        gamma=loss_cfg['matcher']['gamma']
    )
    criter = SetCriterion(
        matcher = Hungarian,
        weight_dict = loss_cfg['SetCriterion']['weight_dict'],
        losses = loss_cfg['SetCriterion']['losses'], 
        alpha = loss_cfg['SetCriterion']['alpha'], 
        gamma=loss_cfg['SetCriterion']['gamma'], 
        eos_coef=loss_cfg['SetCriterion']['eos_coef'], 
        num_classes=data_cfg['num_classes']
    )

    # taining visualization setting
    for epoch in range(training_cfg['epoch']):
        model.train()
        for batch, (data, target) in enumerate(train_dataset, 1):
            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            # data = data.to(cfg['device'])
            
            # forward pass
            data = torch.stack(data).to(cfg['device'])
            target = torch.stack(target).to(cfg['device'])
            print(target)
            target = target.to(cfg['device'])
            
            output = model(data)
            # calculate the loss
            trainloss = criter(output, target)
            loss = sum(trainloss.values())
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()
            
        model.eval()
        for data, target in valid_dataset:
            # data = data
            # forward pass: compute predicted outputs by passing inputs to the model
            data = torch.stack(data).to(cfg['device'])
            
            target = torch.stack(target).to(cfg['device'])
            output = model(data)
            # calculate the loss
            validloss = criter(output, target)
            valloss = sum(validloss)
            # record validation loss

        # if epoch ==training_cfg['save_period']:
        #     wandb.log({"train": trainloss})
        #     wandb.log({"valid": validloss})
        pass
    print(f"training msg : [{epoch:>{training_cfg['epoch']}}/{epoch:>{training_cfg['epoch']}}] | tarin loss : {loss} | valid loss : {valloss }")


    pass


if __name__ == '__main__':
    main()
