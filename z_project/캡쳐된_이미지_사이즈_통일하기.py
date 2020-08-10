import cv2
import numpy as np
import os

imgList = os.listdir("./image")

imgList
path = "./image/"

for i in imgList:
    img = cv2.imread(path + i)
    height, width = img.shape[:2]

    dst = cv2.resize(img, dsize=(64, 64), interpolation=cv2.INTER_AREA)

    cv2.imwrite((path + i), dst)
