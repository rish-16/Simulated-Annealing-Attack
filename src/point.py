import numpy as np
import tensorflow as tf

class Point:
    def __init__(self, og_img):
        self.shape = og_img.shape
        self.colour = [np.random.uniform(0, 1), np.random.uniform(0, 1), np.random.uniform(0, 1)] # RGB Channels
        self.coords = {}
        
        x, y = np.where(og_img <= 1)
        random_idx = np.random.randint(len(x))
        
        self.coords['x'] = x[random_idx]
        self.coords['y'] = y[random_idx]
        
    def superimpose(self, img):
        img_copy = img.copy()
        img_copy[self.coords['x']][self.coords['y']] = self.colour
        
        return img, img_copy