# GoogLeNet 모델로 얼굴분류훈련

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


# 각 train 이미지에 라벨이 잘 붙었지 확인합니다.
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
from keras import Model
from keras import layers
from keras import optimizers     

# Inception 모듈 정의
def inception_module(model, o_1=64, r_3=64, o_3=128, r_5=16, o_5=32, pool=32):
    """
    # Arguments 
    x : 입력이미지
    o_1 : 1x1 convolution 연산 출력값의 채널 수 
    r_3 : 3x3 convolution 이전에 있는 1x1 convolution의 출력값 채널 수
    o_3 : 3x3 convolution 연산 출력값의 채널 수 
    r_5 : 5x5 convolution 이전에 있는 1x1 convolution의 출력값 채널 수 
    o_5 : 5x5 convolution 연산 출력값의 채널 수 
    pool: maxpooling 다음의 1x1 convolution의 출력값 채널 수
    
    # returns
    4 종류의 연산의 결과 값을 채널 방향으로 합친 결과 
    """
    
    model_1 = layers.Conv2D(o_1, 1, padding='same')(model)
    
    model_2 = layers.Conv2D(r_3, 1, padding='same')(model)
    model_2 = layers.Conv2D(o_3, 3, padding='same')(model_2)
    
    model_3 = layers.Conv2D(r_5, 1, padding='same')(model)
    model_3 = layers.Conv2D(o_5, 5, padding='same')(model_3)
    
    model_4 = layers.MaxPooling2D(pool_size=(3, 3), strides=1, padding='same')(model)
    model_4 = layers.Conv2D(pool, 1, padding='same')(model_4)
    
    return layers.concatenate([model_1, model_2, model_3, model_4])

input = layers.Input(shape=(256, 256, 3),  dtype='float32', name='input')

model = layers.Conv2D(64, 7, strides=2, padding='same', name='block1_conv1')(input)
model = layers.MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')(model)
model = layers.BatchNormalization()(model)

model = layers.Conv2D(64, (1,1), strides=1, name='block2_conv1')(model)
model = layers.Conv2D(192, (3,3), strides=1, padding='same', name='block2_conv2')(model)
model = layers.BatchNormalization()(model)
model = layers.MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')(model)

model = inception_module(model, o_1=64, r_3=64, o_3=128, r_5=16, o_5=32, pool=32)
model = inception_module(model, o_1=128, r_3=128, o_3=192, r_5=32, o_5=96, pool=64)
model = layers.MaxPooling2D(pool_size=(3, 3))(model)

model = inception_module(model, o_1=192, r_3=96, o_3=208, r_5=16, o_5=48, pool=64)
model = inception_module(model, o_1=160, r_3=112, o_3=224, r_5=24, o_5=64, pool=64)
model = inception_module(model, o_1=128, r_3=128, o_3=256, r_5=24, o_5=64, pool=64)
model = inception_module(model, o_1=112, r_3=144, o_3=288, r_5=32, o_5=64, pool=64)
model = inception_module(model, o_1=256, r_3=160, o_3=320, r_5=32, o_5=128, pool=128)
model = layers.MaxPooling2D(pool_size=(3, 3))(model)

model = inception_module(model, o_1=256, r_3=160, o_3=320, r_5=32, o_5=128, pool=128)
model = inception_module(model, o_1=384, r_3=192, o_3=384, r_5=48, o_5=128, pool=128)
model = layers.MaxPooling2D(pool_size=(3, 3), strides=1)(model)

model = layers.Dropout(0.4)(model)
model = layers.Dense(5)(model)
output = layers.Activation('softmax')(model)

googlenet = Model(input, output)

callback_list = [keras.callbacks.EarlyStopping(monitor = 'val_acc', patience = 5),
                 keras.callbacks.ModelCheckpoint(filepath='GoogLeNet.h5', monitor = 'val_loss', save_best_only = True)]

googlenet.compile(loss = 'sparse_categorical_crossentropy', 
              optimizer = 'adam',
              metrics = ['acc'])
              
googlenet.summary()

# Model Fit! 제발 잘 됐으면 좋.겠.다.
import time # 훈련할 때마다 time을 가져와서 초기화를 시켜줘야 합니다!
start = time.time() 
history = googlenet.fit(train_image, train_label, epochs = 100, callbacks=callback_list,validation_data = (test_image,test_label))
time = time.time() - start
print("테스트 시 소요 시간(초) : {}".format(time))
print("전체 파라미터 수 : {}".format(sum([arr.flatten().shape[0] for arr in googlenet.get_weights()])))


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
test1 = cv2.imread("./TestIMG/sk/frame5.jpg")
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

googlenet.predict(test1)























