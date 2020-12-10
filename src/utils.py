import numpy as np

class Mask:
    def superimpose(self, image):
        w  = image.shape[0]
        h  = image.shape[1]
        
        r_loc = np.random.randint(0, h)
        c_loc = np.random.randint(0, w)
        
        color = np.random.randint(0, 255) / 255
        image[r_loc][c_loc] = color
        
        return image