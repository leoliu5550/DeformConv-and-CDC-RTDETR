import numpy as np
import matplotlib.pyplot as plt
import cv2
box = np.array([[0.9050, 0.0938, 1.4096, 0.5581],
        [0.1244, 0.5570, 0.4344, 0.6432]])*800
print(box)


# print(box_cxcywh_to_xyxy(box[0]))

path = "/home/lcliu/Documents/EXP_RTDETR/img3.png"
image = cv2.imread(path) 
startpoint = int(box[0][:2])
print(startpoint)
endpoint = int(box[0][2:])
print(endpoint)
color = (255, 0, 0) 
thickness = 2
image = cv2.rectangle(image, startpoint , endpoint,color, thickness) 
