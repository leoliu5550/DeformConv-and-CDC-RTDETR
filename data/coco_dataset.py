"""
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

COCO dataset which returns image_id for evaluation.
Mostly copy-paste from https://github.com/pytorch/vision/blob/13b35ff/references/detection/coco_utils.py
"""
import torch
import torch.utils.data
import torchvision
torchvision.disable_beta_transforms_warning()
from torchvision import datapoints
from torchvision import transforms
from .functional import *
from pycocotools import mask as coco_mask
import yaml
import logging
import logging.config
logging.config.fileConfig('logging.conf')
logger = logging.getLogger(f"coco.{__name__}")



class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, ann_file, # transforms, 
                return_masks,# size, 
                yaml_path,
                remap_mscoco_category=False):
        super(CocoDetection, self).__init__(img_folder, ann_file)
        # self._transforms = transforms
        self.prepare = ConvertCocoPolysToMask(return_masks, remap_mscoco_category)
        self.img_folder = img_folder
        self.ann_file = ann_file
        self.return_masks = return_masks
        self.remap_mscoco_category = remap_mscoco_category
        with open(yaml_path,"r") as file:
            cfg = yaml.safe_load(file)
        self.device = cfg['device']
        # self.size = size
    def __getitem__(self, idx):
        img, target = super(CocoDetection, self).__getitem__(idx)
        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}
        img, target = self.prepare(img, target)

        # ['boxes', 'masks', 'labels']:
        if 'boxes' in target:
            target['boxes'] = datapoints.BoundingBox(
                target['boxes'], 
                format=datapoints.BoundingBoxFormat.XYXY, 
                spatial_size=img.size[::-1]) # h w

        if 'masks' in target:
            target['masks'] = datapoints.Mask(target['masks'])

        # if self._transforms is not None:
        #     img, target = self._transforms(img, target)


        # category2label
        for labels_idx in range(len(target['labels'])):
            target['labels'][labels_idx] = category2label[int(target['labels'][labels_idx])]-1
        
        transform1 = transforms.Compose([transforms.ToTensor()])  #归一化到(0,1)，简单直接除以255
        # print("=-img-=")
        # print(img.size) # (640, 480) 
        logger.debug("#"*20+"before_target"+"#"*20)
        logger.debug(target)
        img, target = resize(transform1(img),target)
        logger.debug("#"*20+"after_target"+"#"*20)
        logger.debug(target)
        # {'boxes': BoundingBox([[385.5300,  60.0300, 600.5000, 357.1900],
        #              [ 53.0100, 356.4900, 185.0400, 411.6800]], 
        # format=BoundingBoxFormat.XYXY, spatial_size=(426, 640)), 'labels': tensor([23, 23]), 'image_id': tensor([25]), 'area': tensor([19686.5977,  2785.8477]), 'iscrowd': tensor([0, 0]), 'orig_size': tensor([640, 426]), 'size': tensor([640, 426])}

        # print(img.shape) # torch.Size([3, 480, 640])
        # C x H x W in [0,1]

        # move data and labels to device
        img = img.to(self.device)
        for key, value in target.items():
            target[key] = target[key].to(self.device)

        return img, target

    def extra_repr(self) -> str:
        s = f' img_folder: {self.img_folder}\n ann_file: {self.ann_file}\n'
        s += f' return_masks: {self.return_masks}\n'
        # if hasattr(self, '_transforms') and self._transforms is not None:
        #     s += f' transforms:\n   {repr(self._transforms)}'
        return s 


def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


class ConvertCocoPolysToMask(object):
    def __init__(self, return_masks=False, remap_mscoco_category=False):
        self.return_masks = return_masks
        self.remap_mscoco_category = remap_mscoco_category

    def __call__(self, image, target):
        w, h = image.size

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]

        anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]

        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        if self.remap_mscoco_category:
            classes = [category2label[obj["category_id"]] - 1 for obj in anno]
        else:
            classes = [obj["category_id"] for obj in anno]
            
        classes = torch.tensor(classes, dtype=torch.int64)

        # image segmetations
        if self.return_masks:
            segmentations = [obj["segmentation"] for obj in anno]
            masks = convert_coco_poly_to_mask(segmentations, h, w)

        keypoints = None
        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.view(num_keypoints, -1, 3)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        if self.return_masks:
            masks = masks[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        if self.return_masks:
            target["masks"] = masks
        target["image_id"] = image_id
        if keypoints is not None:
            target["keypoints"] = keypoints

        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]

        target["orig_size"] = torch.as_tensor([int(w), int(h)])
        target["size"] = torch.as_tensor([int(w), int(h)])
    
        return image, target


# {'boxes': tensor([[260.1875, 163.8438, 505.9375, 276.9062],
#         [ 62.2000, 480.9219, 308.8500, 713.3282],
#         [133.7625, 256.3750, 292.8500, 553.6250],
#         [323.2125, 262.3438, 456.2875, 661.5625],
#         [205.0875, 220.1094, 244.3250, 271.2187],
#         [140.6000, 227.1094, 219.1250, 417.9688],
#         [661.2750, 184.6875, 736.3500, 446.6719],
#         [703.5375, 225.8906, 794.9375, 505.0781],
#         [ 34.8250, 311.3125,  97.0250, 424.9844],
#         [634.2625, 237.3906, 685.7500, 389.6562],
#         [464.9000, 361.9688, 495.2375, 441.8906],
#         [269.5500, 326.3906, 547.8750, 435.4375],
#         [425.4750, 254.0781, 484.3000, 420.1250],
#         [238.7125, 262.8750, 291.8750, 374.9062]]), 'labels': tensor([28, 28,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1]), 'image_id': tensor([525939]), 'area': tensor([14339.5342, 29926.3184, 26330.7852, 36522.7266,  1122.1887,  7303.7803,
#          9558.4521, 13974.3857,  3605.6980,  2998.3481,  1442.3893,  4123.8794,
#          5652.4263,  2603.5833]), 'iscrowd': tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), 'orig_size': tensor([640, 512]), 'size': tensor([800, 800])}


names = {
    0: 'background',
    1: 'person',
    2: 'bicycle',
    3: 'car',
    4: 'motorcycle',
    5: 'airplane',
    6: 'bus',
    7: 'train',
    8: 'truck',
    9: 'boat',
    10: 'traffic light',
    11: 'fire hydrant',
    13: 'stop sign',
    14: 'parking meter',
    15: 'bench',
    16: 'bird',
    17: 'cat',
    18: 'dog',
    19: 'horse',
    20: 'sheep',
    21: 'cow',
    22: 'elephant',
    23: 'bear',
    24: 'zebra',
    25: 'giraffe',
    27: 'backpack',
    28: 'umbrella',
    31: 'handbag',
    32: 'tie',
    33: 'suitcase',
    34: 'frisbee',
    35: 'skis',
    36: 'snowboard',
    37: 'sports ball',
    38: 'kite',
    39: 'baseball bat',
    40: 'baseball glove',
    41: 'skateboard',
    42: 'surfboard',
    43: 'tennis racket',
    44: 'bottle',
    46: 'wine glass',
    47: 'cup',
    48: 'fork',
    49: 'knife',
    50: 'spoon',
    51: 'bowl',
    52: 'banana',
    53: 'apple',
    54: 'sandwich',
    55: 'orange',
    56: 'broccoli',
    57: 'carrot',
    58: 'hot dog',
    59: 'pizza',
    60: 'donut',
    61: 'cake',
    62: 'chair',
    63: 'couch',
    64: 'potted plant',
    65: 'bed',
    67: 'dining table',
    70: 'toilet',
    72: 'tv',
    73: 'laptop',
    74: 'mouse',
    75: 'remote',
    76: 'keyboard',
    77: 'cell phone',
    78: 'microwave',
    79: 'oven',
    80: 'toaster',
    81: 'sink',
    82: 'refrigerator',
    84: 'book',
    85: 'clock',
    86: 'vase',
    87: 'scissors',
    88: 'teddy bear',
    89: 'hair drier',
    90: 'toothbrush'
}


label2category = {
    1: 1,
    2: 2,
    3: 3,
    4: 4,
    5: 5,
    6: 6,
    7: 7,
    8: 8,
    9: 9,
    10: 10,
    11: 11,
    12: 13,
    13: 14,
    14: 15,
    15: 16,
    16: 17,
    17: 18,
    18: 19,
    19: 20,
    20: 21,
    21: 22,
    22: 23,
    23: 24,
    24: 25,
    25: 27,
    26: 28,
    27: 31,
    28: 32,
    29: 33,
    30: 34,
    31: 35,
    32: 36,
    33: 37,
    34: 38,
    35: 39,
    36: 40,
    37: 41,
    38: 42,
    39: 43,
    40: 44,
    41: 46,
    42: 47,
    43: 48,
    44: 49,
    45: 50,
    46: 51,
    47: 52,
    48: 53,
    49: 54,
    50: 55,
    51: 56,
    52: 57,
    53: 58,
    54: 59,
    55: 60,
    56: 61,
    57: 62,
    58: 63,
    59: 64,
    60: 65,
    61: 67,
    62: 70,
    63: 72,
    64: 73,
    65: 74,
    66: 75,
    67: 76,
    68: 77,
    69: 78,
    70: 79,
    71: 80,
    72: 81,
    73: 82,
    74: 84,
    75: 85,
    76: 86,
    77: 87,
    78: 88,
    79: 89,
    80: 90
}

category2label = {v: k for k, v in label2category.items()}