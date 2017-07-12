# this one is trained vs udacity sample data...

import csv
import numpy as np
#from random import shuffle
import cv2
import matplotlib.pyplot as plt
import os

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Convolution2D, Cropping2D, Dropout, MaxPooling2D

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

import matplotlib.pyplot as plt
#from sklearn.utils import shuffle
import random

ignore_frames = [
'43_20_883.jpg',
'43_20_984.jpg',
'43_21_084.jpg', 
'43_21_187.jpg', 
'43_21_288.jpg', 
'43_21_390.jpg',
'43_21_492.jpg',
'43_21_595.jpg',
'43_21_696.jpg',
'43_21_799.jpg',
'43_21_900.jpg',
'43_22_002.jpg',
'43_22_102.jpg',
'43_22_203.jpg',
'43_22_305.jpg',
'43_22_406.jpg',
'43_22_508.jpg',
'43_22_610.jpg',
'43_22_713.jpg',
'43_22_814.jpg',
'43_22_916.jpg',
'43_23_019.jpg',
'43_23_121.jpg',
'43_23_223.jpg',
'43_23_325.jpg',
'43_23_427.jpg',
'43_23_529.jpg',
'43_23_631.jpg',
'43_23_733.jpg',
'43_23_836.jpg',
'43_23_936.jpg',
'43_24_034.jpg',
'43_24_141.jpg',
'43_24_242.jpg'
]

def check_ignore(x):
    for frame in ignore_frames:
        if x.endswith(frame):
            return False
    return True

def add_all_data_log(path):
    lines = []
    with open(path) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:        
            if check_ignore(line[0]):
                lines.append(line)
    
    images = []
    measurements = []
    for line in lines:
        source_path = line[0].strip()
        image = cv2.cvtColor(cv2.imread(source_path), cv2.COLOR_BGR2RGB)
        images.append(image)
        measurement = float(line[3])
        measurements.append(measurement)
        
        # add left camera with steering adjustment
        source_path = line[1].strip()
        image = cv2.cvtColor(cv2.imread(source_path), cv2.COLOR_BGR2RGB)
        images.append(image)
        measurements.append(measurement + 0.2)
        
        # add right camera with steering adjustment
        source_path = line[2].strip()
        image = cv2.cvtColor(cv2.imread(source_path), cv2.COLOR_BGR2RGB)
        images.append(image)
        measurements.append(measurement - 0.2)
    return images, measurements

"""
paths = ['./data/driving_log.csv', 
         './bridge-correction/driving_log.csv']

images = []
measurements = []
for path in paths:
    img, m = add_all_data_log(path)
    images.extend(img[:])
    measurements.extend(m[:])

augmented_images, augmented_measurements = [], []
for image, measurement in zip(images, measurements):
     augmented_images.append(image)
     augmented_measurements.append(measurement)
     augmented_images.append(cv2.flip(image,1))
     augmented_measurements.append(measurement*-1.0)

X_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)

#input_shape = (160, 320, 3)
X_train, y_train = shuffle(X_train, y_train)
"""

# all data...
lines = []
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:        
        if check_ignore(line[0]):
            lines.append(line)

with open('./bridge-correction/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:        
        lines.append(line)

# convert line to multiple lists so it can go in a function callable

def gen_sample(line, steering, is_flip, steer_angle=0.2):
    measurement = float(line[3])
    
    if steering == 0:
        source_path = line[0].strip()
    elif steering == -1:
        # right camera
        source_path = line[2].strip()
        measurement += steering*steer_angle
    elif steering == 1:
        # left camera
        source_path = line[1].strip()
        measurement += steering*steer_angle
    
    image = cv2.cvtColor(cv2.imread(source_path), cv2.COLOR_BGR2RGB)
    if is_flip:
        image = cv2.flip(image, 1)
    
    # can add random adjustment if you wish here for more samples
    return image, measurement

# generate all combinations....
import itertools
sample = [lines, [-1, 0, 1], [True, False]]
samples = list(itertools.product(*sample))

from sklearn.model_selection import train_test_split
import sklearn

def generator(samples, batch_size = 32):
    num_samples = len(samples)    
    while 1: # Loop forever so the generator never terminates
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            
            images = []
            angles = []
            for batch_sample in batch_samples:
                image, angle = gen_sample(*batch_sample)                
                images.append(image)
                angles.append(angle)
            
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

batch_size = 128
train_samples, validation_samples = train_test_split(samples, test_size=0.2)
train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)

model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape = (160, 320, 3))) # normalise image
model.add(Cropping2D(cropping=((70,25), (30,30))))
model.add(Convolution2D(24, 5, 5, subsample=(2,2), activation="relu"))
model.add(Convolution2D(36, 5, 5, subsample=(2,2), activation="relu"))
model.add(Convolution2D(48, 5, 5, subsample=(2,2), activation="relu"))
model.add(Convolution2D(64, 3, 3, activation="relu"))
model.add(Convolution2D(64, 3, 3, activation="relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
#model.add(Dropout(0.3))
model.add(Dense(1))
#model.add(Dropout(0.4))

model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator, 
        samples_per_epoch= len(train_samples), 
        validation_data=validation_generator, 
        nb_val_samples=len(validation_samples), nb_epoch=3)


model.save('model_20170708.h5')

