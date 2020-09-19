import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from sim_ann_attack import SimulatedAnnealingAttack
from point import Point

model = ResNet50(include_top=True, weights="imagenet")

img_path = "../assets/dog2.jpg"
img = image.load_img(img_path, target_size=(224, 224)) # standard resnet input dims
img = image.img_to_array(img) / 255
x = np.expand_dims(img, axis=0)
x = preprocess_input(x)

preds = model.predict(x)
labels = decode_predictions(preds, top=1)
print (labels)

T_max = 1000
T_min = 1
T_delta = 0.01
temps = np.arange(T_max, T_min, T_delta)

all_images = []
all_epsilons = []
all_images.append(x)

def mutate_point(pt):
    random_color = [np.random.uniform(0, 1), np.random.uniform(0, 1), np.random.uniform(0, 1)] # RGB Channels
    pt.colour = random_color
    
    return pt
    
for i in range(len(temps)):
    T = temps[i]
    
    point = Point(x)
    
    img_point = point.superimpose(x)
    
    epsilon = decode_predictions(model.predict(x), top=1)[0][0][-1]
    epsilon_point = decode_predictions(model.predict(img_point), top=1)[0][0][-1]
    
    if epsilon_point < epsilon: # new image lowers confidence more
        mut_point = mutate_point(point)
        img_mut_point = mut_point.superimpose(x)
        epsilon_mut = decode_predictions(model.predict(img_mut_point), top=1)[0][0][-1]
        
        if epsilon_mut < epsilon_point:
            x = img_mut_point
            epsilon = epsilon_mut
        else:
            x = img_point
            epsilon = epsilon_point
        
    elif np.exp(-epsilon / T) < np.random.uniform(0, 1): # not using Boltzmann Constant k
        x = img_point
        
    if i % 50 == 0:
        print ("Current iteration: {} | Temperate: {} | Epsilon: {}".format(i+1, T, epsilon))
        
    all_images.append(x)
    all_epsilons.append(epsilon)
    
plt.tight_layout()
plt.figure(1)
plt.subplot(121)
plt.imshow(all_images[0])
plt.axis("off")
plt.subplot(122)
plt.imshow(all_images[-1])
plt.axis("off")