
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D
from tensorflow.keras.optimizers import Adam
import scipy


train_dir = 'emotion_images/train'
test_dir = 'emotion_images/test'


import glob
import numpy as np
from PIL import Image
filelist = glob.glob('emotion_images/train/*/*')
x = np.array([np.array(Image.open(fname)) for fname in filelist])
x = x/255.
y = []
for i in filelist:
    a = i.split('/')
    y.append(a[2])

import pandas as pd
y_ser = pd.Series(y)

y = pd.get_dummies(y_ser, dtype ='int')


model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(filters = 10, kernel_size=3, activation='relu', input_shape=(48,48,1)),
    tf.keras.layers.Conv2D(filters = 10, kernel_size=3, activation = 'relu'),
    tf.keras.layers.MaxPool2D(pool_size=2, padding='valid'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(7, activation = 'softmax')
])


model.compile(loss = tf.keras.losses.CategoricalCrossentropy(),
              optimizer=tf.keras.optimizers.Adam(),
              metrics = ['accuracy'])



model.fit(x, y, epochs = 100)



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
model.predict(np.expand_dims(neutral_img, axis = 0))