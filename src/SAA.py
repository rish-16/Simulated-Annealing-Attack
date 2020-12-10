import copy
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist
from utils import Mask

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape([len(x_train), 28, 28, 1]) / 255
x_test = x_test.reshape([len(x_test), 28, 28, 1]) / 255

y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

model = load_model("../bin/mnist_cnn.h5")

original = copy.deepcopy(x_test[0])
image = x_test[0]
label = y_test[0].max()

# plt.imshow(original.reshape([28, 28]), cmap="gray")
# plt.title(y_test[0].argmax())
# plt.show()

def normalised_RMSE(img, cpy):
    mse = np.sum(np.square(img - cpy))
    return np.sqrt(mse)

all_confidences = []

T_MIN, T_MAX, T_DELTA = 1, 1000, 1
temperatures = np.arange(T_MAX, T_MIN, -T_DELTA)

for i in range(len(temperatures)):
    t = temperatures[i]
                
    logit = model.predict(np.array([image]))
    initial_confidence = logit.max()
    
    point = Mask()
    image_with_point = point.superimpose(image)
    confidence_with_point = model.predict(np.array([image_with_point])).max()
    print (normalised_RMSE(image, image_with_point), confidence_with_point)
    
    if confidence_with_point < initial_confidence and normalised_RMSE(image, image_with_point) < 0.2:
        new_point = Mask()
        image_with_new_point = new_point.superimpose(image_with_point)
        confidence_with_new_point = model.predict(np.array([image_with_new_point])).max()
        
        if confidence_with_new_point < confidence_with_point and normalised_RMSE(image_with_point, image_with_new_point) < 0.2:
            image = image_with_new_point
            initial_confidence = confidence_with_new_point
        else:
            image = image_with_point
            initial_confidence = confidence_with_point
            
    elif np.exp(-initial_confidence/t) < np.random.uniform(0, 1):
        image = image_with_point
        initial_confidence = confidence_with_point
        
    all_confidences.append(initial_confidence)
        
fig = plt.figure(1)        

plt.subplot(131)
plt.imshow(original.reshape([28, 28]), cmap="gray")
plt.title(model.predict(np.array([original])).argmax())
plt.axis("off")

plt.subplot(132)
plt.imshow(image.reshape([28, 28]), cmap="gray")
plt.title(model.predict(np.array([image])).argmax())
plt.axis("off")

plt.subplot(133)
plt.plot(temperatures, all_confidences, color="red")
plt.xlabel("Temperature")
plt.ylabel("Confidence")

plt.show()