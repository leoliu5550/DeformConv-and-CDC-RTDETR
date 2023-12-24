import torch
from torch.optim import Adam
from torch.utils.data import  DataLoader
import wandb

import yaml
import os
from data import *
from model import *
from model import rtdetr
import numpy as np
import logging
import logging.config
logging.config.fileConfig('logging.conf')
logger = logging.getLogger(f"train.{__name__}")


cfg_path = "model_config.yaml"
with open(cfg_path,"r") as file:
    cfg = yaml.safe_load(file)

# project total config
training_cfg = cfg['train']
data_cfg = cfg['data']
opt_cfg = cfg['optimizer']['adam']
loss_cfg = cfg['LOSS']

def predata():
    def collate_fn(batch):
        return tuple(zip(*batch))
    train_dataset = DataLoader(
        CocoDetection(
                img_folder = data_cfg['train_dataloader']['dataset']['img_folder'],
                ann_file = data_cfg['train_dataloader']['dataset']['ann_file'],
                return_masks = False,
                yaml_path=cfg_path
            ),
        batch_size=data_cfg['train_dataloader']['batch_size'],
        shuffle=data_cfg['train_dataloader']['shuffle'],
        collate_fn=collate_fn
    )
    valid_dataset = DataLoader(
        CocoDetection(
                img_folder = data_cfg['val_dataloader']['dataset']['img_folder'],
                ann_file = data_cfg['val_dataloader']['dataset']['ann_file'],
                return_masks = False,
                yaml_path=cfg_path
            ),
        batch_size=data_cfg['val_dataloader']['batch_size'],
        shuffle=data_cfg['val_dataloader']['shuffle'],
        collate_fn=collate_fn
    )
    return train_dataset,valid_dataset 

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



def main():
    last_loss = np.inf
    trigger_times = 0
    train_dataset, valid_dataset = predata()
    # taining visualization setting
    for epoch in range(training_cfg['epoch']):
        model.train()
        for batch, (data, target) in enumerate(train_dataset, 1):
            # clear the gradients of all optimized variables
            optimizer.zero_grad()                                                

            # forward pass
            data = torch.stack(data) 
            output = model(data)
            
            # calculate the loss
            trainloss = criter(output, target)
            loss = sum(trainloss.values())
            logger.info(f"train loss : {loss}")
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()
            
        model.eval()
        for data, target in valid_dataset:
            # data = data
            # forward pass: compute predicted outputs by passing inputs to the model
            data = torch.stack(data) #.to(cfg['device'])
            
            # target = torch.stack(target).to(cfg['device'])
            output = model(data)
            # calculate the loss
            validloss = criter(output, target)
             
            logger.debug(f"validloss : {validloss}")
            valloss = sum(validloss.values())
            logger.info(f"valid loss : {valloss}")
            # record validation loss

        if epoch == training_cfg['save_period']:

            logger.info(f"loss save at {epoch}")
            torch.save(model,os.path.join(cfg['save_dir'],"last.pt"))
            

            
            
        # Early stop setting
        # if valloss > last_loss:
        #     trigger_times += 1
        #     if trigger_times >= training_cfg['tolerance_period']:
        #         torch.save(model,os.path.join(cfg_path['save_dir'],"earlystop_{epoch}.pt"))
        #         break
        # else:
        #     trigger_times = 0
        # last_loss = valloss
# {epoch:>{training_cfg['epoch']}}/{epoch:>{training_cfg['epoch']}}

        logger.info(f"training msg : [{epoch}/{training_cfg['epoch']}] | tarin loss : {loss} | valid loss : {valloss }")
        dir_path = os.path.join(cfg['train']['save_dir'],f"exp{len(os.listdir(cfg['train']['save_dir']))+1}")
        if not os.path.exists(dir_path) :
            os.makedirs(dir_path) 
        torch.save(model,os.path.join(dir_path,f"checkpoint{epoch}.pt"))
        print(model)

    pass

if __name__ == "__main__":
    main()