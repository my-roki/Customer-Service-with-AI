import cv2
import numpy as np
import os

custlist = os.listdir("./image/")
custlist

for i in custlist:
    path = "./image/" + i + "/"
    imglist = os.listdir(path)
    print(imglist)
    num=1
    
    for j in imglist:
        img = cv2.imread(path + j)  # 이미지 사이즈 통일하기
        height, width = img.shape[:2]
        dst = cv2.resize(img, dsize=(256, 256), interpolation=cv2.INTER_AREA)
        cv2.imwrite((path + j), dst)
        
        src = os.path.join(path, j) # 이미지 이름 바꾸기
        dst = i+ str(num) + '.jpg'
        dst = os.path.join(path, dst)
        os.rename(src, dst)    
        num += 1   
