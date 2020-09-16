import numpy as np
import tensorflow as tf
from point import Point

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