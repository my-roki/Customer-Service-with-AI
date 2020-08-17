#InceptionV3 모델로 얼굴분류훈련

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

# 모델 생성 InceptionV4
import keras
from keras import Model
from keras import layers
from keras import losses
from keras import optimizers     

# Inception 모듈 정의
def inception_module_A(model, o_A1=96, r_A3=64, o_A3=96, r_A5=64, o_A5=96, pool_A=96):
    model_A1 = layers.Conv2D(o_A1, (1,1), strides=1, padding='same')(model)
    
    model_A2 = layers.Conv2D(r_A3, (1,1), strides=1, padding='same')(model)
    model_A2 = layers.Conv2D(o_A3, (3,3), strides=1, padding='same')(model_A2)
    
    model_A3 = layers.Conv2D(r_A5, (1,1), strides=1, padding='same')(model)
    model_A3 = layers.Conv2D(o_A5, (3,3), strides=1, padding='same')(model_A3)
    model_A3 = layers.Conv2D(o_A5, (3,3), strides=1, padding='same')(model_A3)

    model_A4 = layers.AveragePooling2D(pool_size=(3, 3), strides=1, padding='same')(model)
    model_A4 = layers.Conv2D(pool_A, (1,1), strides=1, padding='same')(model_A4)
    
    return layers.concatenate([model_A1, model_A2, model_A3, model_A4])


def inception_module_B(model, o_B1=384, r_B3=192, s_B3 = 224 , o_B3=256, r_B5=192, s_B5=224, o_B5=256, pool_B=128):
    model_B1 = layers.Conv2D(o_B1, (1,1), strides=1, padding='same')(model)
    
    model_B2 = layers.Conv2D(r_B3, (1,1), strides=1, padding='same')(model)
    model_B2 = layers.Conv2D(s_B3, (1,7), strides=1, padding='same')(model_B2)
    model_B2 = layers.Conv2D(o_B3, (7,1), strides=1, padding='same')(model_B2)
    
    model_B3 = layers.Conv2D(r_B5, (1,1), strides=1, padding='same')(model)
    model_B3 = layers.Conv2D(r_B5, (1,7), strides=1, padding='same')(model_B3)
    model_B3 = layers.Conv2D(s_B5, (7,1), strides=1, padding='same')(model_B3)
    model_B3 = layers.Conv2D(s_B5, (1,7), strides=1, padding='same')(model_B3)
    model_B3 = layers.Conv2D(o_B5, (7,1), strides=1, padding='same')(model_B3)
    
    model_B4 = layers.AveragePooling2D(pool_size=(3, 3), strides=1, padding='same')(model)
    model_B4 = layers.Conv2D(pool_B, (1,1), strides=1, padding='same')(model_B4)
    
    return layers.concatenate([model_B1, model_B2, model_B3, model_B4])    
    
        
def inception_module_C(model, o_C1=256, r_C3=384, o_C3=256, r_C5=384, o_C5=448, s_C5=512, t_C5=256, pool_C=256):
    model_C1 = layers.Conv2D(o_C1, (1,1), strides=1, padding='same')(model)
    
    model_C2 = layers.Conv2D(r_C3, (1,1), strides=1, padding='same')(model)
    model_C2_1 = layers.Conv2D(o_C3, (1,3), strides=1, padding='same')(model_C2)
    model_C2_2 = layers.Conv2D(o_C3, (3,1), strides=1, padding='same')(model_C2)
    
    model_C3 = layers.Conv2D(r_C5, (1,1), strides=1, padding='same')(model)
    model_C3 = layers.Conv2D(o_C5, (1,3), strides=1, padding='same')(model_C3)
    model_C3 = layers.Conv2D(s_C5, (3,1), strides=1, padding='same')(model_C3)
    model_C3_1 = layers.Conv2D(t_C5, (1,3), strides=1, padding='same')(model_C3)
    model_C3_2 = layers.Conv2D(t_C5, (3,1), strides=1, padding='same')(model_C3)
    
    model_C4 = layers.AveragePooling2D(pool_size=(3, 3), strides=1, padding='same')(model)
    model_C4 = layers.Conv2D(pool_C, (1,1), strides=1, padding='same')(model_C4)
    
    return layers.concatenate([model_C1, model_C2_1, model_C2_2, model_C3_1, model_C3_2, model_C4])    


def Reduction_A(model, r_RA3=384, r_RA5=224, o_RA5=256, s_RA5=384):
    model_RA1 = layers.Conv2D(r_RA3, (3,3), strides=2, padding='valid')(model)

    model_RA2 = layers.Conv2D(r_RA5, (1,1), strides=1, padding='same')(model)
    model_RA2 = layers.Conv2D(o_RA5, (3,3), strides=1, padding='same')(model_RA2)
    model_RA2 = layers.Conv2D(s_RA5, (3,3), strides=2, padding='valid')(model_RA2)
    
    model_RA3 = layers.MaxPooling2D(pool_size=(3, 3), strides=2, padding='valid')(model)
    
    return layers.concatenate([model_RA1, model_RA2, model_RA3])   


    
def Reduction_B(model, r_RB3=256, o_RB3=384, r_RB5=256, o_RB5=320):
    model_RB1 = layers.Conv2D(r_RB3, (1,1), strides=1, padding='same')(model)
    model_RB1 = layers.Conv2D(o_RB3, (3,3), strides=2, padding='valid')(model_RB1)

    model_RB2 = layers.Conv2D(r_RB5, (1,1), strides=1, padding='same')(model)
    model_RB2 = layers.Conv2D(r_RB5, (1,7), strides=1, padding='same')(model_RB2)
    model_RB2 = layers.Conv2D(o_RB5, (7,1), strides=1, padding='same')(model_RB2)
    model_RB2 = layers.Conv2D(o_RB5, (3,3), strides=2, padding='valid')(model_RB2)
    
    model_RB3 = layers.MaxPooling2D(pool_size=(3, 3), strides=2, padding='valid')(model)
    
    return layers.concatenate([model_RB1, model_RB2, model_RB3])   


input = layers.Input(shape=(256, 256, 3),  dtype='float32', name='input')

model_SL1 = layers.Conv2D(32, (3,3), strides=2, padding='same', name='block1_conv1')(input)
model_SL1 = layers.Conv2D(32, (3,3), strides=1, padding='valid', name='block1_conv2')(model_SL1)
model_SL1 = layers.Conv2D(64, (3,3), strides=1, padding='valid', name='block1_conv3')(model_SL1)

model_SL1_1 = layers.MaxPooling2D(pool_size=(3, 3), strides=2, padding='valid')(model_SL1)
model_SL1_2 = layers.Conv2D(96, (3,3), strides=2, padding='valid', name='block2_conv1')(model_SL1)

model_SL2 = layers.concatenate([model_SL1_1, model_SL1_2])

model_SL2_1 = layers.Conv2D(64, (1,1), strides=1, padding='same', name='block3_conv1')(model_SL2)
model_SL2_1 = layers.Conv2D(96, (3,3), strides=1, padding='same', name='block3_conv2')(model_SL2_1)

model_SL2_2 = layers.Conv2D(64, (1,1), strides=1, padding='same', name='block4_conv1')(model_SL2)
model_SL2_2 = layers.Conv2D(64, (7,1), strides=1, padding='same', name='block4_conv2')(model_SL2_2)
model_SL2_2 = layers.Conv2D(64, (1,7), strides=1, padding='same', name='block4_conv3')(model_SL2_2)
model_SL2_2 = layers.Conv2D(96, (3,3), strides=1, padding='same', name='block4_conv4')(model_SL2_2)

model_SL3 = layers.concatenate([model_SL2_1, model_SL2_2])

model_SL3_1 = layers.Conv2D(192, (3,3), strides=2, padding='valid', name='block2_conv2')(model_SL3)
model_SL3_2 = layers.MaxPooling2D(pool_size=(3, 3), strides=2, padding='valid')(model_SL3)

model = layers.concatenate([model_SL3_1, model_SL3_2])

model = inception_module_A(model, o_A1=96, r_A3=64, o_A3=96, r_A5=64, o_A5=96, pool_A=96)
model = inception_module_A(model, o_A1=96, r_A3=64, o_A3=96, r_A5=64, o_A5=96, pool_A=96)
model = inception_module_A(model, o_A1=96, r_A3=64, o_A3=96, r_A5=64, o_A5=96, pool_A=96)
model = inception_module_A(model, o_A1=96, r_A3=64, o_A3=96, r_A5=64, o_A5=96, pool_A=96)
model = Reduction_A(model, r_RA3=384, r_RA5=224, o_RA5=256, s_RA5=384)

model = inception_module_B(model, o_B1=384, r_B3=192, s_B3 = 224 , o_B3=256, r_B5=192, s_B5=224, o_B5=256, pool_B=128)
model = inception_module_B(model, o_B1=384, r_B3=192, s_B3 = 224 , o_B3=256, r_B5=192, s_B5=224, o_B5=256, pool_B=128)
model = inception_module_B(model, o_B1=384, r_B3=192, s_B3 = 224 , o_B3=256, r_B5=192, s_B5=224, o_B5=256, pool_B=128)
model = inception_module_B(model, o_B1=384, r_B3=192, s_B3 = 224 , o_B3=256, r_B5=192, s_B5=224, o_B5=256, pool_B=128)
model = inception_module_B(model, o_B1=384, r_B3=192, s_B3 = 224 , o_B3=256, r_B5=192, s_B5=224, o_B5=256, pool_B=128)
model = inception_module_B(model, o_B1=384, r_B3=192, s_B3 = 224 , o_B3=256, r_B5=192, s_B5=224, o_B5=256, pool_B=128)
model = inception_module_B(model, o_B1=384, r_B3=192, s_B3 = 224 , o_B3=256, r_B5=192, s_B5=224, o_B5=256, pool_B=128)
model = Reduction_B(model, r_RB3=256, o_RB3=384, r_RB5=256, o_RB5=320)

model = inception_module_C(model, o_C1=256, r_C3=384, o_C3=256, r_C5=384, o_C5=448, s_C5=512, t_C5=256, pool_C=256)
model = inception_module_C(model, o_C1=256, r_C3=384, o_C3=256, r_C5=384, o_C5=448, s_C5=512, t_C5=256, pool_C=256)
model = inception_module_C(model, o_C1=256, r_C3=384, o_C3=256, r_C5=384, o_C5=448, s_C5=512, t_C5=256, pool_C=256)
model = layers.AveragePooling2D(pool_size=(6, 6))(model)
model = layers.Dropout(0.8)(model)

model = layers.Dense(4)(model)
output = layers.Activation('softmax')(model)

InceptionV4 = Model(input, output)

callback_list = [keras.callbacks.EarlyStopping(monitor = 'val_acc', patience = 5), 
                    keras.callbacks.ModelCheckpoint(filepath='InceptionV4.h5', monitor = 'val_loss', save_best_only = True)]

InceptionV4.compile(loss = 'sparse_categorical_crossentropy', 
              optimizer = 'RMSProp',
              metrics = ['acc'])
      
InceptionV4.summary()

# Model Fit! 제발 잘 됐으면 좋.겠.다.
import time # 훈련할 때마다 time을 가져와서 초기화를 시켜줘야 합니다!
start = time.time() 
history = InceptionV4.fit(train_image, train_label, epochs = 100, callbacks=callback_list, validation_data = (test_image,test_label))
time = time.time() - start
print("테스트 시 소요 시간(초) : {}".format(time))
print("전체 파라미터 수 : {}".format(sum([arr.flatten().shape[0] for arr in InceptionV4.get_weights()])))


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



from keras.applications.inception_v3 import InceptionV3, decode_predictions

inceptionv3 = InceptionV3(input_shape=(299,299,3))
inceptionv3.summary()


















