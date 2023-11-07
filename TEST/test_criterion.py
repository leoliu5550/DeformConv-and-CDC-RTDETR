import sys
sys.path.append(".")
import yaml
from model import (SetCriterion,
                HungarianMatcher,
                accuracy)



# class Testcriterion:
#     with open("model_config.yaml","r") as file:
#         cfg = yaml.safe_load(file)
#         cfg = cfg['LOSS']
#     Hungarian = HungarianMatcher(
#         weight_dict = cfg["matcher"]["weight_dict"],
#         use_focal_loss=False, 
#         alpha=0.25, 
#         gamma=2.0
#     )
#     criter = rtdetr_criterion(
#         matcher = Hungarian,
#         weight_dict = cfg["SetCriterion"]["weight_dict"],
#         losses = cfg["SetCriterion"]["losses"], 
#         alpha=0.2, 
#         gamma=2.0, 
#         eos_coef=1e-4, 
#         num_classes=80
#     )
#     pass

# class TestCoCo:
#     with open("model_config.yaml") as file:
#         cfg = yaml.safe_load(file)
#     cfg = cfg['data']['train_dataloader']['dataset']
#     Coco_loader = CocoDetection(
#         img_folder = cfg['img_folder'],
#         ann_file = cfg['ann_file'],
#         # transforms = None,
#         return_masks = False
#     )

#     data = Coco_loader.__getitem__(1)
#     def test_Coco_loader(self):
#         assert self.data[0].shape == torch.Size([3, 426, 640])


with open("model_config.yaml","r") as file:
    cfg = yaml.safe_load(file)
    cfg = cfg['LOSS']
Hungarian = HungarianMatcher(
    weight_dict = cfg["matcher"]["weight_dict"],
    use_focal_loss=False, 
    alpha=0.25, 
    gamma=2.0
)
criter = SetCriterion(
    matcher = Hungarian,
    weight_dict = cfg["SetCriterion"]["weight_dict"],
    losses = cfg["SetCriterion"]["losses"], 
    alpha=0.2, 
    gamma=2.0, 
    eos_coef=1e-4, 
    num_classes=80
)

# DATA and Target
from data import *
with open("model_config.yaml") as file:
    cfg = yaml.safe_load(file)
cfg = cfg['data']['train_dataloader']['dataset']
Coco_loader = CocoDetection(
    img_folder = cfg['img_folder'],
    ann_file = cfg['ann_file'],
    # transforms = None,
    return_masks = False,
    yaml_path = "model_config.yaml"
)
data = Coco_loader.__getitem__(1)
data, target =resize(data[0],data[1])

# model output
from model.rtdetr import rtdetr
model = rtdetr('model_config.yaml')
image = data
output = model(data)


# func test
# ans = criter(output,[target])
print("-"*80)
print(target)
print("="*80)
# print(ans)
# acc = accuracy(output,data[1])
# print(acc)
# for id,rows in enumerate(ans):
#     # print()
#     print(id,rows)