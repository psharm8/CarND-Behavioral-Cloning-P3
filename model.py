import csv
import cv2
from PIL import Image
from math import ceil
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D

samples = []
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

train_samples, validation_samples = train_test_split(samples, test_size=0.2)

def process_image(img):
    croppped = img[50:130,:,:] # Crop to see only the road
    color = cv2.cvtColor(croppped, cv2.COLOR_BGR2RGB)
    return color

def filename(path):
    return path.split('\\')[-1]


def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                steering_center = float(batch_sample[3])
                # create adjusted steering measurements for the side camera images
                correction = 0.2
                steering_left = steering_center + correction
                steering_right = steering_center - correction

                # read in images from center, left and right cameras
                path = "./data/IMG/"
                img_center = process_image(cv2.imread(path + filename(batch_sample[0])))
                img_left = process_image(cv2.imread(path + filename(batch_sample[1])))
                img_right = process_image(cv2.imread(path + filename(batch_sample[2])))

                images.extend([img_center, img_left, img_right])
                angles.extend([steering_center, steering_left, steering_right])

            X_train = np.array(images)
            y_train = np.array(angles)
            yield shuffle(X_train, y_train)

# Generators for training and validation
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

ch, row, col = 3, 80, 320  # Trimmed image format
EPOCHS = 5
train_steps = ceil(len(train_samples)/32)
validation_steps = ceil(len(validation_samples)/32)
print(train_steps)
print(validation_steps)
model = Sequential()
model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=(row, col, ch)))
# model.add(Cropping2D(cropping=((50,25), (0,0)), input_shape=(3,160,320)))
model.add(Conv2D(24, (5,5), activation='relu', strides=(2, 2)))
model.add(Conv2D(36, (5,5), activation='relu', strides=(2, 2)))
model.add(Conv2D(48, (5,5), activation='relu', strides=(2, 2)))
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator, steps_per_epoch= train_steps, validation_data=validation_generator, validation_steps=validation_steps, epochs=EPOCHS)
model.save('model.h5')
exit()