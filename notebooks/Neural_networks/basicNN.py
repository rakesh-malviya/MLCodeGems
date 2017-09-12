import numpy as np

class Dense:
    def __init__(self,layer_size,input_size,act_func):
        self.W = np.random.normal(loc=0,scale=0.01,size=(layer_size,input_size))
        self.b = np.zeros(shape=(layer_size,1))
        self.h = None
        self.z = None






