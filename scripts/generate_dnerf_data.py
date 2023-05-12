import cv2
import os
import numpy as np

img_list = []
mask_list = []
img_path = "/workspace/sdfstudio/data/davis/train/images"
mask_path = "/workspace/sdfstudio/data/davis/train/masks"

for _, _, files in os.walk(img_path):
    for file in files:
        img_name = os.path.join(img_path, file)
        img = cv2.imread(img_name)
        img_list.append(img)

for _, _, files in os.walk(mask_path):
    for file in files:
        mask_name = os.path.join(mask_path, file)
        mask = cv2.imread(mask_name)
        mask_list.append(mask)

for i in range(len(img_list)):
    dynamic = cv2.cvtColor(img_list[i], cv2.COLOR_BGR2BGRA)
    dynamic[(mask_list[i]==255)[:,:,0]] = 0
    cv2.imwrite('/workspace/sdfstudio/data/davis/dytrain/images/frame_{0:0>5}.png'.format(i+1), dynamic)
