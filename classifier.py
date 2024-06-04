
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D
from tensorflow.keras.optimizers import Adam
import scipy
import pandas as pd
import glob
import numpy as np
from PIL import Image
import random
from tensorflow.keras.callbacks import EarlyStopping


train_dir = 'emotion_images/train'
test_dir = 'emotion_images/test'


#train_data
train_files = glob.glob('emotion_images/train/*/*')

train_files = random.sample(train_files, len(train_files))

x_train = np.array([np.array(Image.open(fname)) for fname in train_files])
x_train = x_train/255.
y_train = []
for i in train_files:
    a = i.split('/')
    y_train.append(a[2])

y_train = pd.Series(y_train)
y_train = pd.get_dummies(y_train, dtype ='int').values


#test_data
test_files = glob.glob('emotion_images/test/*/*')
test_files = random.sample(test_files, len(test_files))
x_test = np.array([np.array(Image.open(fname)) for fname in test_files])
x_test = x_test/255.
y_test = []
for i in test_files:
    a = i.split('/')
    y_test.append(a[2])
y_test = pd.Series(y_test)
y_test = pd.get_dummies(y_test, dtype ='int').values


cb = EarlyStopping(monitor = 'val_loss', patience = 5)

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(filters = 64, kernel_size=3, activation='relu', input_shape=(48,48,1)),
    tf.keras.layers.Conv2D(filters = 64, kernel_size=3, activation = 'relu'),
    tf.keras.layers.MaxPool2D(pool_size=2, padding='valid'),
    tf.keras.layers.Conv2D(filters = 32, kernel_size=3, activation='relu'),
    tf.keras.layers.Conv2D(filters = 32, kernel_size=3, activation = 'relu'),
    tf.keras.layers.MaxPool2D(pool_size=2, padding='valid'),
    tf.keras.layers.Conv2D(filters = 32, kernel_size=3, activation='relu'),
    tf.keras.layers.Conv2D(filters = 32, kernel_size=3, activation = 'relu'),
    tf.keras.layers.MaxPool2D(pool_size=2, padding='valid'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(32, activation = 'relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(7, activation = 'softmax')
])


model.compile(loss = tf.keras.losses.CategoricalCrossentropy(),
              optimizer=tf.keras.optimizers.Adam(),
              metrics = ['accuracy'])

model.summary()

model.fit(x_train, y_train, batch_size = 32, validation_data=(x_test, y_test), epochs = 100, shuffle = True, callbacks=[cb])

model.save('emotion_classifier2.h5')
'''
sad_img = Image.open('emotion_images/me/sad.JPG')
from PIL import ImageOps
sad_img = ImageOps.grayscale(sad_img)
sad_img = sad_img.resize((48,48))
sad_img = np.array(sad_img)
sad_img = sad_img/255.
model.predict(np.expand_dims(sad_img, axis = 0))


happy_img = Image.open('emotion_images/me/happy.JPG')
happy_img = ImageOps.grayscale(happy_img)
happy_img = happy_img.resize((48,48))
happy_img = np.array(happy_img)
happy_img = happy_img/255.
model.predict(np.expand_dims(happy_img, axis = 0))



neutral_img = Image.open('emotion_images/me/neutral.JPG')
neutral_img = ImageOps.grayscale(neutral_img)
neutral_img = neutral_img.resize((48,48))
neutral_img = np.array(neutral_img)
neutral_img = neutral_img/255.
model.predict(np.expand_dims(neutral_img, axis = 0))'''