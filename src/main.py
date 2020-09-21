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

T_max = 1000
T_min = 1
T_delta = 0.01
temps = np.arange(T_max, T_min, -T_delta)

all_images = []
all_epsilons = []
all_images.append(img)

def mutate_point(pt):
    random_color = [np.random.uniform(0, 1), np.random.uniform(0, 1), np.random.uniform(0, 1)] # RGB Channels
    pt.colour = random_color
    
    return pt
    
def get_confidence(img):
    x = np.expand_dims(img, axis=0)
    x = preprocess_input(img)
    pred = model.predict(x)
    label = decode_predictions(pred, top=1)
    confidence = label[0][0][-1]
    
    return confidence
    
def superimpose(pt, img):
    img_copy = img.copy()
    img_copy[pt.coords['x']][pt.coords['y']] = pt.colour
    
    return img_copy
    
print ("Poisoning image...")
for i in range(len(temps)):
    T = temps[i]
    
    point = Point(img)
    
    img_point = superimpose(point, img)
    
    epsilon = get_confidence(img)
    epsilon_point = get_confidence(img_point)
    
    if epsilon_point < epsilon: # new image lowers confidence more
        mut_point = mutate_point(point)
        img_mut_point = superimpose(mut_point, img)
        epsilon_mut = get_confidence(img_mut_point)
        
        if epsilon_mut < epsilon_point:
            img = img_mut_point
            epsilon = epsilon_mut
        else:
            img = img_point
            epsilon = epsilon_point
        
    elif np.exp(-epsilon / T) < np.random.uniform(0, 1): # not using Boltzmann Constant k
        img = img_point
        
    if i % 50 == 0:
        print ("Current iteration: {} | Temperate: {} | Epsilon: {}".format(i+1, T, epsilon))
        
    all_images.append(img)
    all_epsilons.append(epsilon)