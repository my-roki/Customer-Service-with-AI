#image dataset 만들기 파일 경로는 수정해야함 일단 희조랑 나랑 카테고리에 넣고해봣음

import numpy as np
import glob
from PIL import Image
from sklearn.model_selection import train_test_split

caltech_dir = "./img/"
categories = ['yang', 'hui']
nb_classes = len(categories)

image_w = 64
image_h = 64

pixels = image_h * image_w * 3

X = []
Y = []

for idx, cat in enumerate(categories):
    label = [0 for i in range(nb_classes)]
    label[idx] = 1
    
    image_dir = caltech_dir + "/" + cat
    files = glob.glob(image_dir + "/*.jpg")
    
    for i, f in enumerate(files):
        img = Image.open(f)
        img = img.convert("RGB")
        img = img.resize((image_w, image_h))
        data = np.asarray(img)
        data = data / 255.0
        
        X.append(data)
        Y.append(label)
        
        
X = np.array(X)
Y = np.array(Y)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y)
xy = (X_train, X_test, Y_train, Y_test)

np.save("./img_data.npy", xy)
        
        
