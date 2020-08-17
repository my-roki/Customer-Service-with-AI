# Siamese Neural Network
print("Hello Atom!")

import os
faces_dir = "./image/" 

from keras.preprocessing.image import load_img, img_to_array
import numpy as np
train_image, train_label = [], []
test_image, test_label = [], []

#faces_dir 아래의 하위 디렉터리 목록을 가져옵니다.
#각 하위 디렉터리는 대상의 이미지를 담습니다.
subfolders = sorted([f.path for f in os.scandir(faces_dir) if f.is_dir()])
#하위 디렉터리 목록을 대상으로 반복문을 실행합니다.
#idx를 대상자의 ID로 사용합니다.

for idx, folder in enumerate(subfolders):
    for file in sorted(os.listdir(folder)):
        img = load_img(folder+"/"+file)
        img = img_to_array(img).astype('float32')/255
        img = img.reshape(img.shape[0], img.shape[1], 3)
        if idx < 2: #인덱스 번호가 0,1인 사람(희조 창록)을 훈련 데이터로 씁니다.
            train_image.append(img)
            train_label.append(idx)
        else: #인덱스 번호가 2,3인 사람(성균 양현)을 테스트 데이터로 씁니다.
            test_label.append(idx-2)

train_image = np.array(train_image)
test_image = np.array(test_image)
train_label = np.array(train_label)
test_label = np.array(test_label)

train_image.shape
test_image.shape

import matplotlib.pyplot as plt # 데이터가 잘 분리 됐는지 확인하는 작업입니다.
subject_idx = 1
fig, ((ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9)) = plt.subplots(3,3, figsize=(5,5))
subject_img_idx = np.where(train_label == subject_idx)[0].tolist()

for i, ax in enumerate([ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8,ax9]):
    img = train_image[subject_img_idx[i]]
    img = np.squeeze(img)
    ax.imshow(img)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    
plt.tight_layout()
plt.show()


# 모델생성
import keras
from keras import Model
from keras import layers
from keras import losses
from keras import optimizers
from keras.models import Sequential, Input    

def Basic_Architecture(input_shape): # 이곳에는 아무 모델이나 들어가도 됩니다. 시헙삼아 Basic_Architecture(막 만든거) 넣었습니다.
                                     # Sequential로 작성해도 되지만 작업적 한계 때문에 Funtional으로 작성했습니다. Inception모델 때문입니다ㅠ
    input = layers.Input(shape = input_shape,  dtype='float32')

    model = layers.Conv2D(32, (3,3), activation = 'relu', padding='same', name='block1_conv1')(input)
    model = layers.MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')(model)
    model = layers.Conv2D(64, (3,3), activation = 'relu', padding='same', name='block2_conv1')(model)
    model = layers.MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')(model)
    model = layers.Conv2D(128, (3,3), activation = 'relu', padding='same', name='block3_conv1')(model)
    model = layers.MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')(model)
    model = layers.Flatten()(model)
    model = layers.Dense(128)(model)

    output = layers.Activation('sigmoid')(model) #sigmoid함수는 결과값을 0-1사이로 표현해준다. 확률을 나타낼 때 요긴하게 쓰는 듯.
    
    basic = Model(input,output)
    
    return basic

# 서로 다른 이미지가 같은 모델을 지나가면서 동시에 컨볼루션 레이어가 가중치를 공유한다는 점이 흥미롭습니다.
input_shape = train_image.shape[1:]
shared_network = Basic_Architecture(input_shape)

input_top = Input(shape = input_shape)
input_bottom = Input( shape = input_shape)

output_top = shared_network(input_top)
output_bottom = shared_network(input_bottom)

#유클리드 거리함수입니다. 두 이미지가 얼마나 닮았는지 판별하는 함수입니다.
from keras import backend as K 
def euclidian_distance(vectors):
    vector1, vector2 = vectors
    sum_square = K.sum(K.square(vector1 - vector2), axis=1, keepdims=True)
    
    return K.sqrt(K.maximum(sum_square, K.epsilon()))

#distance = 유클리드 거리함수로 두 이미지의 거리가 얼마나 되는지 출력해주는 변수입니다.
#거리가 가까울수록 두 이미지는 일치하다고 판단합니다.
from keras.layers import Lambda
distance = Lambda(euclidian_distance, output_shape=(1,))([output_top, output_bottom])

basic_model = Model(inputs=[input_top, input_bottom], outputs=distance) #모델형성

basic_model.summary() #모델 형성 요약

# train_image와 test_image에서 이미지 배열 쌍과 레이블을 생성하는 함수
import random
def create_pairs(X,Y, num_classes):
    pairs, labels = [], []
    # 각 클래스의 이미지를 가리키는 인덱스 배열
    class_idx = [np.where(Y==i)[0] for i in range (num_classes)]
    #이미지가 가장 적은 클래스의 이미지 개수
    min_images = min(len(class_idx[i]) for i in range(num_classes)) - 1
    for c in range(num_classes):
        for n in range(min_images):
            #양성 쌍을 생성한다
            img1 = X[class_idx[c][n]]
            img2 = X[class_idx[c][n+1]]
            pairs.append((img1, img2))
            labels.append(1)
            #음성 쌍을 생성한다
            #현재 클래스 리스트c를 제외한 나머지 클래스 리스트
            neg_list = list(range(num_classes))
            neg_list.remove(c)
            #나머지 클래스 리스트에서 무작위로 하나 골라 음성 쌍에 사용한다
            neg_c = random.sample(neg_list,1)[0]
            img1 = X[class_idx[c][n]]
            img2 = X[class_idx[neg_c][n]]
            pairs.append((img1, img2))
            labels.append(0)
    return np.array(pairs), np.array(labels)


#새로운 training, test 셋 만들기            
training_pairs, training_labels = create_pairs(train_image, train_label, len(np.unique(train_label)))
test_pairs, test_labels = create_pairs(test_image, test_label, len(np.unique(test_label)))

#대조손실함수
def contrastive_loss(Y_true, D):
    margin = 1
    return K.mean(Y_true*K.square(D)+(1 - Y_true)*K.maximum((margin-D),0))

#callbacks과 ModelCheckpoint설정    
callback_list = [keras.callbacks.EarlyStopping(monitor = 'acc', patience = 5),
                keras.callbacks.ModelCheckpoint(filepath='model.h5', monitor = 'loss', save_best_only = True)]
    
basic_model.compile(loss = contrastive_loss, optimizer = 'adam',  metrics = ['acc'])
        
# Model Fit! 제발 잘 됐으면 좋.겠.다.
history = basic_model.fit([training_pairs[:,0], training_pairs[:,1]], training_labels, epochs = 10, callbacks=callback_list)

#테스트 이미지 쌍으로 얼마나 잘 작동하는 지 확인
idx1, idx2 = 2, 60
img1 = np.expand_dims(test_image[idx1], axis = 0)
img2 = np.expand_dims(test_image[idx2], axis = 0)


fig, (ax1,ax2) = plt.subplots(1,2, figsize=(10,7))
ax1.imshow(np.squeeze(img1))
ax2.imshow(np.squeeze(img2))

for ax in [ax1,ax2]:
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    
dissimilarity = basic_model.predict([img1, img2])[0][0]
fig.suptitle("Dissimilarity Score = {:.3f}".format(dissimilarity), size=30)
plt.tight_layout()
plt.show()
    
# 몇몇 이미지 쌍을 골라 결과를 나타낸 것
for i in range(5):
    for n in range(0,2):
        fig, (ax1,ax2) = plt.subplots(1,2, figsize=(7,5))
        img1 = np.expand_dims(test_pairs[i*20+n,0], axis = 0)
        img2 = np.expand_dims(test_pairs[i*20+n,1], axis = 0)
        dissimilarity = basic_model.predict([img1, img2])[0][0]
        img1, img2 = np.squeeze(img1), np.squeeze(img2)
        ax1.imshow(img1)
        ax2.imshow(img2)

        for ax in [ax1,ax2]:
            ax.grid(False)
            ax.set_xticks([])
            ax.set_yticks([])
            
        plt.tight_layout()
        fig.suptitle("Dissimilarity Score = {:.3f}".format(dissimilarity), size=20)
    
plt.show()






