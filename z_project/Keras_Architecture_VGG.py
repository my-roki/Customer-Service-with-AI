#VGG 모델로 얼굴분류훈련

print("Hello Atom!")

import os 
import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.model_selection import train_test_split

categories = os.listdir("./image")  #각 이미지에 라벨을 붙여줍니다(0부터 차례대로 시작합니다.)
categories
len(categories)

X = []
Y = []

# 이미지 데이터 전처리
for i in categories:
    path = "./image/" + i + "/"
    imglist = os.listdir(path)
    label = [categories.index(i)]
    print(imglist)
    print(label)

    for j in imglist:
        path2 = path + j
        img = cv2.imread(path2)
        b, g, r = cv2.split(img)    
        img2 = cv2.merge([r,g,b])  
        img2 = np.asarray(img2)
        img2 = img2 / 255.0
        
        X.append(img2)
        Y.append(label)
        
X = np.array(X)
Y = np.array(Y)
Y = Y.flatten()
X.shape
Y.shape


#train, test(validation) set 구분합니다.
train_image, test_image, train_label , test_label = train_test_split(X, Y)

train_image.shape
len(train_image)

test_image.shape
len(test_label)

train_label.shape
len(train_label)

test_label.shape
len(test_label)


# 각 train_image에 train_label이 잘 붙었지 확인합니다.
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_image[i], cmap=plt.cm.binary)
    plt.xlabel(categories[train_label[i]])
plt.show()

# 모델 생성 VGG16
import keras
from keras import layers
from keras import models
from keras import optimizers     

model = models.Sequential()
model.add(layers.Input(shape = (256, 256, 3),  dtype='float32', name='input'))

model.add(layers.Conv2D(64, (3,3), activation = 'relu', padding='same', name='block1_conv1'))
model.add(layers.Conv2D(64, (3,3), activation = 'relu', padding='same', name='block1_conv2'))
model.add(layers.MaxPooling2D((2,2)))

model.add(layers.Conv2D(128, (3,3), activation = 'relu', padding ='same', name='block2_conv1'))
model.add(layers.Conv2D(128, (3,3), activation = 'relu', padding ='same', name='block2_conv2'))
model.add(layers.MaxPooling2D((2,2)))

model.add(layers.Conv2D(256, (3,3), activation = 'relu', padding='same', name='block3_conv1'))
model.add(layers.Conv2D(256, (3,3), activation = 'relu', padding='same', name='block3_conv2'))
model.add(layers.Conv2D(256, (3,3), activation = 'relu', padding='same', name='block3_conv3'))
model.add(layers.MaxPooling2D((2,2)))

model.add(layers.Conv2D(512, (3,3), activation = 'relu', padding='same', name='block4_conv1'))
model.add(layers.Conv2D(512, (3,3), activation = 'relu', padding='same', name='block4_conv2'))
model.add(layers.Conv2D(512, (3,3), activation = 'relu', padding='same', name='block4_conv3'))
model.add(layers.MaxPooling2D((2,2)))

model.add(layers.Conv2D(512, (3,3), activation = 'relu', padding='same', name='block5_conv1'))
model.add(layers.Conv2D(512, (3,3), activation = 'relu', padding='same', name='block5_conv2'))
model.add(layers.Conv2D(512, (3,3), activation = 'relu', padding='same', name='block5_conv3'))
model.add(layers.MaxPooling2D((2,2)))

model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(4, activation='softmax'))

callback_list = [keras.callbacks.EarlyStopping(monitor = 'val_acc', patience = 5), 
                 keras.callbacks.ModelCheckpoint(filepath='VGG16.h5', monitor = 'val_loss', save_best_only = True)]

model.compile(loss = 'sparse_categorical_crossentropy', 
              optimizer = 'adam',
              metrics = ['acc'])

model.summary()


# Model Fit! 제발 잘 됐으면 좋.겠.다.
import time # 훈련할 때마다 time을 가져와서 초기화를 시켜줘야 합니다!
start = time.time() 
history = model.fit(train_image, train_label, epochs = 100, callbacks=callback_list, validation_data = (test_image,test_label))
time = time.time() - start
print("테스트 시 소요 시간(초) : {}".format(time))
print("전체 파라미터 수 : {}".format(sum([arr.flatten().shape[0] for arr in model.get_weights()])))


# 모델 훈련이 잘 되었는지 그래프로 확입힙니다.     
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and Validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and Validation loss')
plt.legend()

plt.show()

# 새로운 이미지를 가져와서 예측
test1 = cv2.imread("./TestIMG/z_NotNingen/frame64.jpg")
test1.shape

#cv2 bgr -> rgb로 조정
b, g, r = cv2.split(test1)    
test1 = cv2.merge([r,g,b])  

# 시각화
plt.figure()
plt.imshow(test1)
plt.show()

# 훈련모델에 맞게 전처리
height, width = test1.shape[:2]
test1 = cv2.resize(test1, dsize=(256, 256), interpolation=cv2.INTER_AREA)
test1 = np.expand_dims(test1, axis=0)
test1 = test1.astype(float)/255
test1.shape

model.predict(test1)























