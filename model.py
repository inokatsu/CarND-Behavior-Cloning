import csv
import cv2
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
import pandas as pd
import os
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout
from keras.layers.convolutional import Convolution2D
from keras.optimizers import Adam
from matplotlib import pylab as plt

BATCH_SIZE = 20


def flip_image(image, angle):
    random_flip = np.random.randint(2)
    if random_flip == 0:
        image = cv2.flip(image, 1)
        angle = angle * -1.0
    return image, angle


def select_image(batch_sample):
    select_lcr = np.random.randint(3)
    if select_lcr == 0:
        name = './data/IMG/' + batch_sample['left'].split('/')[-1]
        image = cv2.imread(name)
        angle = float(batch_sample['steering']) + 0.2
    elif select_lcr == 1:
        name = './data/IMG/' + batch_sample['center'].split('/')[-1]
        image = cv2.imread(name)
        angle = float(batch_sample['steering'])
    else:
        name = './data/IMG/' + batch_sample['right'].split('/')[-1]
        image = cv2.imread(name)
        angle = float(batch_sample['steering']) - 0.2
    return image, angle


def random_brightness(image):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    rate = 0.3 + np.random.uniform()
    image[::2] = image[::2] * rate
    return cv2.cvtColor(image, cv2.COLOR_HSV2RGB)


def data_augment(batch_sample):
    image, angle = select_image(batch_sample)  # select from left, center, right images randomly
    image, angle = flip_image(image, angle)  # flip image randomly
    image = random_brightness(image)  # change image brightness randomly
    return image, angle


def crop(image):
    return image[60:-25, :, :]


def resize(image):
    return cv2.resize(image, (200, 66), interpolation=cv2.INTER_AREA)


def blur(image):
    return cv2.GaussianBlur(image, (3, 3), 0)


def rgb2yuv(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2YUV)


def preprocess(image):
    image = crop(image)  # crop image
    image = resize(image)  # resize image
    image = blur(image)     # blur image
    image = rgb2yuv(image)  # change color space
    return image


def batch_generator(samples, batch_size=BATCH_SIZE):
    images = np.zeros([batch_size, 66, 200, 3])
    angles = np.zeros(batch_size)

    while 1:  # Loop forever so the generator never terminates
        i = 0
        for index in np.random.permutation(samples):
            image, angle = data_augment(index)
            images[i] = preprocess(image)
            angles[i] = angle
            i += 1
            if i == batch_size:
                break
        yield sklearn.utils.shuffle(images, angles)


def build_model():
    """
    Modified NVIDIA model
    """
    model = Sequential()
    model.add(Lambda(lambda x: x / 127.5 - 1.0, input_shape=(66, 200, 3)))
    model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation='relu'))
    model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation='relu'))
    model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation='relu'))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(Flatten())
    model.add(Dropout(0.6))
    model.add(Dense(100))
    model.add(Dropout(0.6))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))
    model.summary()
    return model


lines = []
with open('./data/driving_log.csv') as csvfile:
    reader = csv.DictReader(csvfile)
    for line in reader:
        steering = float(line['steering'])
        if steering == 0.0 and np.random.uniform(0, 1) < 0.99:
            continue
        elif abs(steering) < 0.1 and np.random.uniform(0, 1) < 0.8:
            continue
        else:
            lines.append(line)

train_samples, validation_samples = train_test_split(lines, test_size=0.2)

train_generator = batch_generator(train_samples)
validation_generator = batch_generator(validation_samples)

model = build_model()
model.compile(loss='mse', optimizer=Adam(lr=0.0001))

samples_per_epoch = (len(train_samples)//BATCH_SIZE)*BATCH_SIZE
model.fit_generator(train_generator,
                    samples_per_epoch=samples_per_epoch,
                    validation_data=validation_generator,
                    nb_val_samples=len(validation_samples),
                    nb_epoch=3, verbose=1)

model.save('model.h5')
