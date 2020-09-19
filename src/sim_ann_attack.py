import numpy as np
from point import Point

class SimulatedAnnealingAttack:
    def __init__(self, model, image, logits, target_label, T_max=1000, T_min=1, T_delta=0.01):
        self.model = model
        self.image = image
        self.logits = logits
        self.target_label = target_label
        
        self.T_max = T_max
        self.T_min = T_min
        self.T_delta = T_delta
        
        self.gamma  = 0.01 # similarity thresholding for NRMSE
        self.theta = 1 # epsilon thresholding
        
    def nrmse(self, og_img, new_img):
        mse = np.sqrt(np.mean(np.square(og_img - new_img)))
        norm_err = mse / (og_img.max() - new_img.min())
        
        return norm_err
        
    def get_confidence_score(self, img):
        return self.model.predict(img)
        
    def simulated_annealing_iter(self):
        shape = self.image.shape
        
        random_point = Point(self.image)
        new_img = random_point.superimpose()
        
        model_conf = self.get_confidence_score(new_img)
        
    def simulated_annealing(self):
        temps = np.arange(self.T_max, self.T_min, -self.T_delta)
        
        for i in range(len(temps)):
            T = temps[i] # current temperature
            
            point = Point(self.image)
            
            image_with_point = point.superimpose(self.image)
            epsilon_point = self.get_confidence_score(image_with_point) # compare with logits
            
             