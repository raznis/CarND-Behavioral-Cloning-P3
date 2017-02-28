import os
import csv
import random

from keras.callbacks import ModelCheckpoint
from keras.layers import Lambda, Cropping2D, Convolution2D, Activation, Flatten, Dropout, Dense, MaxPooling2D
from keras.models import Sequential
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

import cv2
import numpy as np

import matplotlib.pyplot as plt

DATA_FILE_PREFIX = "./data/"
SIDE_CAMERA_OFFSET = 0.25

TRAIN_NETWORK = True
DOWNSAMPLE_CENTER_PROBABILITY = 0.15 # 0.15

def save_model(filename):
    model.save(filename+".h5")

samples = []

with open(DATA_FILE_PREFIX + 'driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        if(line[0]!="center"):  # skipping the first line
            if not float(line[3]) == 0 or random.random() < DOWNSAMPLE_CENTER_PROBABILITY:
                samples.append(line)

angles = np.array([float(x[3]) for x in samples])


def show_histogram(data, title):
    plt.hist(data, bins=30)
    plt.title(title)
    plt.show()


#show_histogram(angles, "steering angles in downsampled training data")


train_samples, validation_samples = train_test_split(samples, test_size=0.2) # test_size=0.2

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


def random_hsv_changes(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    if(random.random() < 0.3):  #TODO - consider lowering this.
        brighness_alpha = np.random.uniform(0.7,1.1)
        brighness_beta = random.randint(-64, 64)
        v = hsv[:,:,2]
        v = brighness_alpha * v + brighness_beta
        hsv[:, :, 2] = v.astype('uint8')

        sat_alpha = np.random.uniform(0.7, 1.1)
        sat_beta = random.randint(-64, 64)
        v = hsv[:, :, 1]
        v = sat_alpha * v + sat_beta
        hsv[:, :, 1] = v.astype('uint8')

        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        return rgb
    if random.random() < 0.45: # darken a polygon
        w, h, _ = hsv.shape
        x1, y1 = random.randint(0, w), random.randint(0, h)
        x2, y2 = random.randint(x1, w), random.randint(y1, h)
        for i in range(x1,x2):
            for j in range(y1,y2):
                hsv[i,j,2] = int(hsv[i,j,2] * 0.5)
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        return rgb
    return image


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
                # perform brightness and saturation changes
                image = random_hsv_changes(image)
                images.append(image)
                angles.append(angle)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield shuffle(X_train, y_train)





# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=128)
validation_generator = generator(validation_samples, batch_size=128)

#ch, row, col = 3, 80, 320  # Trimmed image format

model = Sequential()
model.add(Cropping2D(cropping=((65,25), (0,0)), input_shape=(160,320,3)))
# Preprocess incoming data, centered around zero with small standard deviation
model.add(Lambda(lambda x: x/127.5 - 1.))#,
                 # input_shape=(ch, row, col),
                 # output_shape=(ch, row, col)))
# apply a 5x5 convolution with 64 output filters on a 256x256 image:
model.add(Convolution2D(24, 5, 5))#, subsample=(2, 2)))
#model.add(MaxPooling2D(border_mode='valid'))
model.add(Activation('relu'))
#model.add(Dropout(0.5))

model.add(Convolution2D(36, 5, 5, subsample=(2, 2)))
model.add(Activation('relu'))

model.add(Convolution2D(48, 5, 5, subsample=(2, 2)))
model.add(Activation('relu'))

model.add(Convolution2D(64, 3, 3))
model.add(Activation('relu'))

model.add(Flatten())
#model.add(Dropout(0.5))

model.add(Dense(100))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(50))
model.add(Activation('relu'))
model.add(Dense(10))
model.add(Activation('relu'))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
checkpointer = ModelCheckpoint(filepath="model.h5", verbose=1, save_best_only=True)

if TRAIN_NETWORK:
    history = model.fit_generator(train_generator,
                        samples_per_epoch=len(train_samples),
                        validation_data=validation_generator,
                        nb_val_samples=len(validation_samples),
                        nb_epoch=30, callbacks=[checkpointer])
    ### print the keys contained in the history object
    print(history.history.keys())

    ### plot the training and validation loss for each epoch
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.show()

    model.save("model.h5")


#visualizing generated data
angles = []
# for i in range(100):
#     angles.extend(next(train_generator)[1])

#show_histogram(angles, "histogram of generated training samples")


# current validation loss is 0.02282