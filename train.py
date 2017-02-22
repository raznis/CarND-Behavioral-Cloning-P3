import os
import csv
import random

from keras.layers import Lambda, Cropping2D, Convolution2D, Activation, Flatten, Dropout, Dense
from keras.models import Sequential
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

import cv2
import numpy as np
import sklearn

DATA_FILE_PREFIX = "./data/"
SIDE_CAMERA_OFFSET = 0.25


def save_model(filename):
    model.save(filename+".h5")

samples = []
with open(DATA_FILE_PREFIX + 'driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        if(line[0]!="center"):
            samples.append(line)


train_samples, validation_samples = train_test_split(samples, test_size=0.2)

def random_flip(image, angle):
    rand_flip = random.randint(0, 1)
    if (rand_flip == 1):
        image = cv2.flip(image, 1)
        angle = -angle
    return image, angle


def random_image_direction(batch_sample):
    index = random.randint(0, 2)
    name = DATA_FILE_PREFIX + 'IMG/' + batch_sample[index].split('/')[-1]
    image = cv2.imread(name)
    center_angle = float(batch_sample[3])
    offset = 0.0
    if index == 1:
        offset = SIDE_CAMERA_OFFSET
    elif index == 2:
        offset = -SIDE_CAMERA_OFFSET
    angle = center_angle + offset
    return angle, image


def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            angles = []
            for batch_sample in batch_samples:
                # randomly choose center,left,right
                angle, image = random_image_direction(batch_sample)
                # randomly flip image and angle
                image, angle = random_flip(image, angle)
                images.append(image)
                angles.append(angle)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield shuffle(X_train, y_train)





# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

ch, row, col = 3, 80, 320  # Trimmed image format

model = Sequential()
model.add(Cropping2D(cropping=((55,25), (0,0)), input_shape=(160,320,3)))
# Preprocess incoming data, centered around zero with small standard deviation
model.add(Lambda(lambda x: x/127.5 - 1.))#,
                 # input_shape=(ch, row, col),
                 # output_shape=(ch, row, col)))
# apply a 5x5 convolution with 64 output filters on a 256x256 image:
model.add(Convolution2D(24, 5, 5, border_mode='same'))
model.add(Activation('relu'))

model.add(Convolution2D(36, 5, 5, subsample=(2, 2)))
model.add(Activation('relu'))

model.add(Convolution2D(48, 5, 5, subsample=(2, 2)))
model.add(Activation('relu'))

model.add(Convolution2D(64, 3, 3))
model.add(Activation('relu'))

model.add(Convolution2D(64, 3, 3))
model.add(Activation('relu'))

model.add(Flatten())
model.add(Dropout(0.5))

model.add(Dense(100))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(50))
model.add(Activation('relu'))
model.add(Dense(10))
model.add(Activation('relu'))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
history = model.fit_generator(train_generator,
                    samples_per_epoch=len(train_samples),
                    validation_data=validation_generator,
                    nb_val_samples=len(validation_samples),
                    nb_epoch=3)
model.save("model.h5")

