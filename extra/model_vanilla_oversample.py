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


lines = []
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:        
        if check_ignore(line[0]):
            lines.append(line)

images = []
measurements = []
for line in lines:
    source_path = line[0]
    image = cv2.imread(source_path)
    images.append(image)
    measurement = float(line[3])
    measurements.append(measurement)

augmented_images, augmented_measurements = [], []
for image, measurement in zip(images, measurements):
    augmented_images.append(image)
    augmented_measurements.append(measurement)
    augmented_images.append(cv2.flip(image,1))
    augmented_measurements.append(measurement*-1.0)

# oversample
oversample_lines = []
with open('./data/driving_log_oversample.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        oversample_lines.append(line)

oversample_images, oversample_measurements = [], []
for line in oversample_lines:
    source_path = line[0]
    image = cv2.imread(source_path)
    oversample_images.append(image)
    measurement = float(line[3])
    oversample_measurements.append(measurement)

# add bridge to overwrite removed ones
# adding better turning
# adding aggressive turning...
bridge_lines = []
with open('./data/driving_log_bridge.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        bridge_lines.append(line)
        
with open('./turn-no-lines/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        bridge_lines.append(line)

with open('./aggressive-turn2/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        bridge_lines.append(line)

with open('./aggressive-turn/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        bridge_lines.append(line)

with open('./aggressive-turn3/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        bridge_lines.append(line)


bridge_images, bridge_measurements = [], []
for line in bridge_lines:
    source_path = line[0]
    image = cv2.imread(source_path)
    bridge_images.append(image)
    bridge_images.append(image)
    measurement = float(line[3])
    bridge_measurements.append(measurement)
    bridge_measurements.append(measurement)
    

X_train = np.array(augmented_images + oversample_images + bridge_images)
y_train = np.array(augmented_measurements + oversample_measurements + bridge_measurements)

#input_shape = (160, 320, 3)
X_train, y_train = shuffle(X_train, y_train)

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
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=3)


model.save('model_vanilla_oversample.h5')

