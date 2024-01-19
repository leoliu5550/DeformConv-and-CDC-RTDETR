'''
by lyuwenyu
'''
import time 
import json
import datetime
import os
import torch 

import requests
from src.misc import dist
from src.data import get_coco_api_from_dataset

from .solver import BaseSolver
from .det_engine import train_one_epoch, evaluate

import logging
import logging.config
logging.config.fileConfig('logging.conf')
logtracker = logging.getLogger(f"train.trainning.{__name__}")
logvalidtracker = logging.getLogger(f"train.vaild.{__name__}")

import wandb,yaml
with open("configs/rtdetr/include/optimizer.yml") as file:
    cfg = yaml.safe_load(file)

if cfg['names']== None:
    wandb.init(
        mode="disabled",
        # set the wandb project where this run will be logged
        project="RTDETR_Refactor_COCO",
        name = cfg['names'],
        # # track hyperparameters and run metadata
        config=cfg
    )
else:
    wandb.init(
        # set the wandb project where this run will be logged
        project="RTDETR_Refactor_COCO",
        name = cfg['names'],
        # # track hyperparameters and run metadata
        config=cfg
    )
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
            
class DetSolver(BaseSolver):
    
    def fit(self, ):
        # print("Start training")
        logtracker.debug("Start training")
        self.train()

        args = self.cfg 
        
        n_parameters = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        # wandb.log({"n_parameters":n_parameters})
    
        wandb.config.n_parameters = n_parameters
 
        logtracker.debug(f"number of params: { n_parameters}")

        base_ds = get_coco_api_from_dataset(self.val_dataloader.dataset)
        # best_stat = {'coco_eval_bbox': 0, 'coco_eval_masks': 0, 'epoch': -1, }
        best_stat = {'epoch': -1, }
        token = "NWBdEgLUPbCLNcs48EaiDBU3jxDIyWBcBwFFDooW3GJ"
        url = 'https://notify-api.line.me/api/notify'
        headers = {
            'Authorization': 'Bearer ' + token    # 設定權杖
        }
        with open("configs/rtdetr/include/optimizer.yml", 'r') as file:
            linecfg = yaml.safe_load(file)['names']
        try:
            start_time = time.time()
            for epoch in range(self.last_epoch + 1, args.epoches):
                if dist.is_dist_available_and_initialized():
                    self.train_dataloader.sampler.set_epoch(epoch)
                
                train_stats,loss_dict = train_one_epoch(
                        self.model, 
                        self.criterion, 
                        self.train_dataloader, 
                        self.optimizer, 
                        self.device, 
                        epoch,
                        args.clip_max_norm, 
                        print_freq=args.log_step, 
                        ema=self.ema, 
                        scaler=self.scaler
                    )
                wandb_prefixlogs(loss_dict)
                self.lr_scheduler.step()
                wandb.log({"learning rate":self.lr_scheduler})
                
                if self.output_dir:
                    checkpoint_paths = [self.output_dir / 'checkpoint.pth']
                    # extra checkpoint before LR drop and every 100 epochs
                    if (epoch + 1) % args.checkpoint_step == 0:
                        checkpoint_paths.append(self.output_dir / f'checkpoint{epoch:04}.pth')
                    for checkpoint_path in checkpoint_paths:
                        dist.save_on_master(self.state_dict(epoch), checkpoint_path)

                module = self.ema.module if self.ema else self.model
                # here to add valid loss record
                test_stats, coco_evaluator,valid_loss_dict = evaluate(
                    module,
                    self.criterion, 
                    self.postprocessor, 
                    self.val_dataloader, 
                    base_ds, 
                    self.device, 
                    self.output_dir
                )
                wandb_prefixlogs(valid_loss_dict,train=False)
                print("test_stats")
                print(test_stats)
                logvalidtracker.debug(f"valid test_stats \n{test_stats}")
                logvalidtracker.debug(f"valid test_stats data type\n{type(test_stats)}")
                logvalidtracker.debug(f"valid coco_evaluator \n{coco_evaluator}")
                logvalidtracker.debug(f"valid coco_evaluator data type\n{type(coco_evaluator)}")
                # TODO 
                for k in test_stats.keys():
                    if k in best_stat:
                        best_stat['epoch'] = epoch if test_stats[k][0] > best_stat[k] else best_stat['epoch']
                        best_stat[k] = max(best_stat[k], test_stats[k][0])
                    else:
                        best_stat['epoch'] = epoch
                        best_stat[k] = test_stats[k][0]
                # print('best_stat: ', best_stat)
                logvalidtracker.debug(f"\n best_stat: \n{best_stat}")

                log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                            **{f'test_{k}': v for k, v in test_stats.items()},
                            'epoch': epoch,
                            'n_parameters': n_parameters}
                
                logtracker.debug(f"\n train and test log_stats: \n{log_stats}")
                if self.output_dir and dist.is_main_process():
                    with (self.output_dir / "log.txt").open("a") as f:
                        f.write(json.dumps(log_stats) + "\n")

                    # for evaluation logs
                    if coco_evaluator is not None:
                        (self.output_dir / 'eval').mkdir(exist_ok=True)
                        if "bbox" in coco_evaluator.coco_eval:
                            filenames = ['latest.pth']
                            if epoch % 50 == 0:
                                filenames.append(f'{epoch:03}.pth')
                            for name in filenames:
                                torch.save(coco_evaluator.coco_eval["bbox"].eval,
                                        self.output_dir / "eval" / name)

            total_time = time.time() - start_time
            total_time_str = str(datetime.timedelta(seconds=int(total_time)))
            print('Training time {}'.format(total_time_str))
            logtracker.debug(f'Training time = {total_time_str}')



            msg = f'\nTASK {linecfg} is finished \nTraining time = {total_time_str}'
            data = {
                'message': msg    # 設定要發送的訊息
            }
            data = requests.post(url, headers=headers, data=data)   # 使用 POST 方法
        except Exception as error:
            
            msg = f'\nTASK {linecfg} is failled \nTraining time = {total_time_str}'
            data = {
                'message': msg    # 設定要發送的訊息
            }
            data = requests.post(url, headers=headers, data=data)   # 使用 POST 方法


    def val(self, ):
        self.eval()

        base_ds = get_coco_api_from_dataset(self.val_dataloader.dataset)
        
        module = self.ema.module if self.ema else self.model
        test_stats, coco_evaluator = evaluate(module, self.criterion, self.postprocessor,
                self.val_dataloader, base_ds, self.device, self.output_dir)
        logtracker.info("\nvalid coco_evaluator part\n{coco_evaluator}")
        logtracker.info("\nvalid test_stats part\n{test_stats}")
        if self.output_dir:
            dist.save_on_master(coco_evaluator.coco_eval["bbox"].eval, self.output_dir / "eval.pth")
        
        return
