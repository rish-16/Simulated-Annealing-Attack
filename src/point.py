import numpy as np
import tensorflow as tf

class Point:
    def __init__(self, og_img):
        self.shape = og_img.shape
        self.coord = (np.random.rand(self.shape))
        self.colour = [np.random.uniform(0, 1), np.random.uniform(0, 1), np.random.uniform(0, 1)] # RGB Channels
        
    def superimpose(self, img):
        img_copy = img.copy()
        img_copy[self.coord] = self.value
        