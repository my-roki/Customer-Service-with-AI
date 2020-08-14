
### part 0

from tensorflow.keras import Input, layers, models
from tensorflow.keras import optimizers, regularizers
from tensorflow.keras.applications import VGG16

import numpy as np


### part 1

input_tensor = Input(shape=(128, 128, 3), dtype='float32', name='input')

# vgg16 모델 불러오기
pre_trained_vgg = VGG16(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
pre_trained_vgg.trainable = False
pre_trained_vgg.summary()

# vgg16 밑에 레이어 추가
model = models.Sequential()
model.add(pre_trained_vgg)
model.add(layers.Flatten())
model.add(layers.Dense(4096, kernel_regularizer = regularizers.l1_l2(l1=0.001,l2=0.001),activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(2048, kernel_regularizer = regularizers.l1_l2(l1=0.001,l2=0.001),activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1024, kernel_regularizer = regularizers.l1_l2(l1=0.001,l2=0.001),activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(4, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer=optimizers.RMSprop(lr=1e-4), metrics=['acc'])

model.summary()


### part 2

X_train, X_test, y_train, y_test = np.load('2ndtrial.npy', allow_pickle=True)

print(X_train.shape)

X_train = X_train.astype(float) / 255
X_test = X_test.astype(float) / 255

history = model.fit(X_train, y_train, batch_size = 16, epochs = 10,  validation_data = (X_test, y_test))

score = model.evaluate(X_test, y_test)


### part 3

from tensorflow.keras.preprocessing import image
import numpy as np

img = image.load_img('frame12.jpg', target_size=(128, 128))
img = image.img_to_array(img)
img = np.expand_dims(img, axis=0)
img = img.astype(float) / 255
model.predict(img)

img = image.load_img('frame17.jpg', target_size=(128, 128))
img = image.img_to_array(img)
img = np.expand_dims(img, axis=0)
img = img.astype(float) / 255
model.predict(img)

img = image.load_img('frame594.jpg', target_size=(128, 128))
img = image.img_to_array(img)
img = np.expand_dims(img, axis=0)
img = img.astype(float) / 255
model.predict(img)

img = image.load_img('frame23.jpg', target_size=(128, 128))
img = image.img_to_array(img)
img = np.expand_dims(img, axis=0)
img = img.astype(float) / 255
model.predict(img)


model.save("model.h5")

del model


### part 4

from tensorflow.keras.models import Sequential, load_model

model = load_model('model.h5')

img = image.load_img('frame12.jpg', target_size=(128, 128))
img = image.img_to_array(img)
img = np.expand_dims(img, axis=0)
img = img.astype(float) / 255
model.predict(img)

img = image.load_img('frame17.jpg', target_size=(128, 128))
img = image.img_to_array(img)
img = np.expand_dims(img, axis=0)
img = img.astype(float) / 255
model.predict(img)

img = image.load_img('frame594.jpg', target_size=(128, 128))
img = image.img_to_array(img)
img = np.expand_dims(img, axis=0)
img = img.astype(float) / 255
model.predict(img)

img = image.load_img('frame23.jpg', target_size=(128, 128))
img = image.img_to_array(img)
img = np.expand_dims(img, axis=0)
img = img.astype(float) / 255
model.predict(img)









