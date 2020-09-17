import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from point import Point
from pprint import pprint
import matplotlib.pyplot as plt

class SimulatedAnnealingAttack:
    def __init__(self, model, image, true_label, target_label):
        self.model = model
        self.image = image
        self.true_label = true_label
        self.target_label = target_label
        
        self.gamma  = 0.01 # similarity thresholding
        
    def nrmse(self, og_img, new_img):
        mse = np.sqrt(np.mean(np.square(og_img - new_img)))
        norm_err = mse / (og_img.max() - new_img.min())
        
        return norm_err
        
    def get_confidence_score(self, img):
        return self.model.predict(img)
        
    def simulated_annealing(self):
        shape = self.image.shape
        
        random_point = Point()

model = ResNet50(include_top=True, weights="imagenet")

img_path = "../assets/elephant.jpg"
img = image.load_img(img_path, target_size=(224, 224)) # standard resnet input dims
img = image.img_to_array(img)
img = np.expand_dims(img, axis=0)
img = preprocess_input(img)

preds = model.predict(img)
label = decode_predictions(preds, top=1)

sim_attack = SimulatedAnnealingAttack(model, img, label, 34)