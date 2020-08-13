print("Hello Atom!")

import os, shutil

original_dataset_dir = "./image"

base_dir = "./customers/fpht_small"
train_dir = os.path.join(base_dir,'train')
validation_dir = os.path.join(base_dir,'validation')
test_dir = os.path.join(base_dir,'test')

try:
    os.mkdir(base_dir)
    os.mkdir(train_dir)
    os.mkdir(validation_dir)
    os.mkdir(test_dir)
except Exception as e: 
    pass


############################################################################
# 이 부분 for루프 구문으로 묶어야 할 듯
train_hui_dir = os.path.join(train_dir,'hui')
train_rok_dir = os.path.join(train_dir,'rok')
train_sk_dir = os.path.join(train_dir,'sk')
train_yang_dir = os.path.join(train_dir,'yang')

try:    
    os.mkdir(train_rok_dir)
    os.mkdir(train_hui_dir)
    os.mkdir(train_sk_dir)
    os.mkdir(train_yang_dir)
except Exception as e: 
    pass


validation_hui_dir = os.path.join(validation_dir,'hui')
validation_rok_dir = os.path.join(validation_dir,'rok')
validation_sk_dir = os.path.join(validation_dir,'sk')
validation_yang_dir = os.path.join(validation_dir,'yang')

try:
    os.mkdir(validation_hui_dir)
    os.mkdir(validation_rok_dir)
    os.mkdir(validation_sk_dir)
    os.mkdir(validation_yang_dir)
except Exception as e: 
    pass


test_hui_dir = os.path.join(test_dir,'hui')
test_rok_dir = os.path.join(test_dir,'rok')
test_sk_dir = os.path.join(test_dir,'sk')
test_yang_dir = os.path.join(test_dir,'yang')

try:
    os.mkdir(test_hui_dir)
    os.mkdir(test_rok_dir)
    os.mkdir(test_sk_dir)
    os.mkdir(test_yang_dir)
except Exception as e: 
    pass


fnames = ['rok{}.jpg'.format(i) for i in range(1,151)]  # 이 부분도 랜덤 이미지 추출지 좋지 않을까?
fnames
for fname in fnames:
    src = os.path.join(original_dataset_dir,"rok/" + fname)
    dst = os.path.join(train_rok_dir,fname) 
    shutil.copyfile(src, dst)

fnames = ['rok{}.jpg'.format(i) for i in range(151,201)] # 이 부분도 랜덤 이미지 추출지 좋지 않을까?
fnames  
for fname in fnames:
    src = os.path.join(original_dataset_dir,"rok/" + fname)
    dst = os.path.join(validation_rok_dir,fname) 
    shutil.copyfile(src, dst)

fnames = ['rok{}.jpg'.format(i) for i in range(201,251)] # 이 부분도 랜덤 이미지 추출지 좋지 않을까?
fnames  
for fname in fnames:
    src = os.path.join(original_dataset_dir,"rok/" + fname)
    dst = os.path.join(test_rok_dir,fname) 
    shutil.copyfile(src, dst)


fnames = ['hui{}.jpg'.format(i) for i in range(1,101)]  # 이 부분도 랜덤 이미지 추출지 좋지 않을까?
fnames
for fname in fnames:
    src = os.path.join(original_dataset_dir,"hui/" + fname)
    dst = os.path.join(train_hui_dir,fname) 
    shutil.copyfile(src, dst)

fnames = ['hui{}.jpg'.format(i) for i in range(101,151)] # 이 부분도 랜덤 이미지 추출지 좋지 않을까?
fnames  
for fname in fnames:
    src = os.path.join(original_dataset_dir,"hui/" + fname)
    dst = os.path.join(validation_hui_dir,fname) 
    shutil.copyfile(src, dst)

fnames = ['hui{}.jpg'.format(i) for i in range(151,200)] # 이 부분도 랜덤 이미지 추출지 좋지 않을까?
fnames  
for fname in fnames:
    src = os.path.join(original_dataset_dir,"hui/" + fname)
    dst = os.path.join(test_hui_dir,fname) 
    shutil.copyfile(src, dst)


fnames = ['sk{}.jpg'.format(i) for i in range(1,121)]  # 이 부분도 랜덤 이미지 추출지 좋지 않을까?
fnames
for fname in fnames:
    src = os.path.join(original_dataset_dir,"sk/" + fname)
    dst = os.path.join(train_sk_dir,fname) 
    shutil.copyfile(src, dst)

fnames = ['sk{}.jpg'.format(i) for i in range(121,181)] # 이 부분도 랜덤 이미지 추출지 좋지 않을까?
fnames  
for fname in fnames:
    src = os.path.join(original_dataset_dir,"sk/" + fname)
    dst = os.path.join(validation_sk_dir,fname) 
    shutil.copyfile(src, dst)

fnames = ['sk{}.jpg'.format(i) for i in range(181,245)] # 이 부분도 랜덤 이미지 추출지 좋지 않을까?
fnames  
for fname in fnames:
    src = os.path.join(original_dataset_dir,"sk/" + fname)
    dst = os.path.join(test_sk_dir,fname) 
    shutil.copyfile(src, dst)
    
    
fnames = ['yang{}.jpg'.format(i) for i in range(1,121)]  # 이 부분도 랜덤 이미지 추출지 좋지 않을까?
fnames
for fname in fnames:
    src = os.path.join(original_dataset_dir,"yang/" + fname)
    dst = os.path.join(train_yang_dir,fname) 
    shutil.copyfile(src, dst)

fnames = ['yang{}.jpg'.format(i) for i in range(121,181)] # 이 부분도 랜덤 이미지 추출지 좋지 않을까?
fnames  
for fname in fnames:
    src = os.path.join(original_dataset_dir,"yang/" + fname)
    dst = os.path.join(validation_yang_dir,fname) 
    shutil.copyfile(src, dst)

fnames = ['yang{}.jpg'.format(i) for i in range(181,240)] # 이 부분도 랜덤 이미지 추출지 좋지 않을까?
fnames  
for fname in fnames:
    src = os.path.join(original_dataset_dir,"yang/" + fname)
    dst = os.path.join(test_yang_dir,fname) 
    shutil.copyfile(src, dst)    
    
################################################################################


from keras import layers
from keras import models     

model = models.Sequential()
model.add(layers.Conv2D(32, (3,3), activation = 'relu', input_shape=(256, 256, 3)))
model.add(layers.MaxPool2D((2,2)))
model.add(layers.Conv2D(64, (3,3), activation = 'relu'))
model.add(layers.MaxPool2D((2,2)))
model.add(layers.Conv2D(128, (3,3), activation = 'relu'))
model.add(layers.MaxPool2D((2,2)))
model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(4, activation='softmax'))

model.summary()

from keras import optimizers
model.compile(loss = 'sparse_categorical_crossentropy', 
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])

from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
                    train_dir,
                    target_size = (256,256),
                    batch_size = 20,
                    class_mode = 'binary')
validation_generator = test_datagen.flow_from_directory(
                    validation_dir,
                    target_size = (256,256),
                    batch_size = 20,
                    class_mode = 'binary')
                    
for data_batch, labels_batch in train_generator:
    print('배치 데이터 크기: ', data_batch.shape)
    print('배치 레이블 크기: ', labels_batch.shape)
    break

history = model.fit_generator(
            train_generator,
            steps_per_epoch = 20,
            epochs = 10,
            validation_data = validation_generator,
            validation_steps = 20)
            

import matplotlib.pyplot as plt

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label = 'Training acc')
plt.plot(epochs, val_acc, 'b', label = 'Validation acc')
plt.title('Training and Validtion accuracy')
plt.legend()
plt.figure()

plt.plot(epochs, loss, 'bo', label = 'Training loss')
plt.plot(epochs, val_loss, 'b', label = 'Validation loss')
plt.title('Training and Validtion loss')
plt.legend()

plt.show()


datagen = ImageDataGenerator(
      rotation_range=40,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest')

# 이미지 전처리 유틸리티 모듈
from keras.preprocessing import image

fnames = sorted([os.path.join(train_rok_dir, fname) for fname in os.listdir(train_rok_dir)])

# 증식할 이미지 선택합니다
img_path = fnames[3]

# 이미지를 읽고 크기를 변경합니다
img = image.load_img(img_path, target_size=(64, 64))

# (64, 64, 3) 크기의 넘파이 배열로 변환합니다
x = image.img_to_array(img)

# (1, 64, 64, 3) 크기로 변환합니다
x = x.reshape((1,) + x.shape)

# flow() 메서드는 랜덤하게 변환된 이미지의 배치를 생성합니다.
# 무한 반복되기 때문에 어느 지점에서 중지해야 합니다!
i = 0
for batch in datagen.flow(x, batch_size=1):
    plt.figure(i)
    imgplot = plt.imshow(image.array_to_img(batch[0]))
    i += 1
    if i % 4 == 0:
        break

plt.show()


model = models.Sequential()
model.add(layers.Conv2D(32, (3,3), activation = 'relu', input_shape=(256, 256, 3)))
model.add(layers.MaxPool2D((2,2)))
model.add(layers.Conv2D(64, (3,3), activation = 'relu'))
model.add(layers.MaxPool2D((2,2)))
model.add(layers.Conv2D(128, (3,3), activation = 'relu'))
model.add(layers.MaxPool2D((2,2)))
model.add(layers.Flatten())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(4, activation='softmax'))

model.summary()

model.compile(loss='sparse_categorical_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])



train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,)

# 검증 데이터는 증식되어서는 안 됩니다!
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
                    train_dir,
                    target_size = (256,256),
                    batch_size = 20,
                    class_mode = 'binary')

train_generator

validation_generator = test_datagen.flow_from_directory(
                    validation_dir,
                    target_size = (256,256),
                    batch_size = 20,
                    class_mode = 'binary')
                    
history = model.fit_generator(
            train_generator,
            steps_per_epoch = 20,
            epochs = 10,
            validation_data = validation_generator,
            validation_steps = 10)



acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()

test_generator = test_datagen.flow_from_directory(
                    test_dir,
                    target_size = (256,256),
                    batch_size = 20,
                    class_mode = 'binary')
import cv2
import numpy as np
import matplotlib.pyplot as plt

test1 = cv2.imread("./customers/fpht_small/test/rok/rok201.jpg")
test1.shape
test1 = np.expand_dims(test1, axis=0)
test1.shape
model.predict(test1)


test2= cv2.imread("./customers/fpht_small/test/sk/sk201.jpg")
test2.shape
test2 = np.expand_dims(test2, axis=0)
test2.shape
model.predict(test2)


test3= cv2.imread("./customers/fpht_small/test/yang/yang224.jpg")
test3.shape
test3 = np.expand_dims(test3, axis=0)
test3.shape
model.predict(test3)


test4= cv2.imread("./customers/fpht_small/test/hui/hui189.jpg")
test4.shape
test4 = np.expand_dims(test4, axis=0)
test4.shape
model.predict(test4)


test5= cv2.imread("./customers/fpht_small/test/rok/rok238.jpg")
test5.shape
test5 = np.expand_dims(test5, axis=0)
test5.shape
model.predict(test5)


test6= cv2.imread("./customers/fpht_small/test/rok/rok250.jpg")
test6.shape
test6 = np.expand_dims(test6, axis=0)
test6.shape
model.predict(test6)


test7= cv2.imread("./customers/fpht_small/test/yang/yang230.jpg")
test7.shape
test7 = np.expand_dims(test7, axis=0)
test7.shape
model.predict(test7)


test8= cv2.imread("./customers/fpht_small/test/yang/yang183.jpg")
test8.shape
test8 = np.expand_dims(test8, axis=0)
test8.shape
p_8 = model.predict(test8)
p_8
np.argmax(p_8[0])








