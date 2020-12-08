import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Dropout, Flatten, Input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape([len(x_train), 28, 28, 1]) / 255
x_test = x_test.reshape([len(x_test), 28, 28, 1]) / 255

y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

model = Sequential()
model.add(Input(shape=[28, 28, 1]))
model.add(Conv2D(32, (3,3), activation="relu"))
model.add(MaxPool2D((2,2)))
model.add(Dropout(0.5))
model.add(Conv2D(64, (3,3), activation="relu"))
model.add(MaxPool2D((2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation="relu"))
model.add(Dense(256, activation="relu"))
model.add(Dense(10, activation="softmax"))
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy", "mse"])

model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10, verbose=1, batch_size=64)
model.save("../bin/mnist_cnn.h5")