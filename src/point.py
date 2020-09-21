import numpy as np
import tensorflow as tf

class Point:
    def __init__(self, og_img):
        self.shape = og_img.shape
        self.colour = [np.random.uniform(0, 1), np.random.uniform(0, 1), np.random.uniform(0, 1)] # RGB Channels
        self.coords = {}
        
        random_idx = np.random.randint(len(og_img))
        
        self.coords['x'] = random_idx
        self.coords['y'] = random_idx