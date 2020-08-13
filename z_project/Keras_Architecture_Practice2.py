import numpy as np
from sklearn.model_selection import train_test_split
import os
import cv2
import matplotlib.pyplot as plt

categories = os.listdir(caltech_dir)
categories
categories[1]

len(categories)

width, height = 256 , 256

X = []
Y = []

for i in categories:
    path = "./image/" + i + "/"
    imglist = os.listdir(path)
    label = [len(i)]
    
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
X.shape
Y.shape



train_image, test_image, train_label , test_label = train_test_split(X, Y)

train_label.flatten()


train_image.shape
train_image.squeeze().shape





len(train_image)

test_image.shape
len(test_label)


plt.figure()
plt.imshow(train_image[0])
plt.colorbar()
plt.grid(False)
plt.show()

plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_image[i], cmap=plt.cm.binary)
    #plt.xlabel(categories[train_label[i]])
plt.show()


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
model.add(layers.Dense(5, activation='softmax'))

from keras import optimizers
model.compile(loss = 'sparse_categorical_crossentropy', 
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])

model.summary()

history = model.fit(train_image, train_label, epochs = 10)
            
import matplotlib.pyplot as plt

acc = history.history['acc']
loss = history.history['loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'b', label = 'Training acc')
plt.title('Training  accuracy')
plt.legend()
plt.figure()

plt.plot(epochs, loss, 'b', label = 'Training loss')
plt.title('Training and Validtion loss')
plt.legend()

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
model.add(layers.Dense(5, activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])

model.summary()

history = model.fit(train_image, train_label, epochs = 10)

acc = history.history['acc']
loss = history.history['loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'b', label='Training acc')
plt.title('Training accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'b', label='Training loss')
plt.title('Training loss')
plt.legend()

plt.show()


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








