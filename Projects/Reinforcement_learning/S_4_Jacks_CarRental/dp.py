import numpy as np
from env import Environment
class DynProg:
    def __init__(self):
        self.env = Environment()
        self.matsize = self.env.max_car + 1
        self.stateValue = np.zeros((self.matsize+1,self.matsize+1))
        self.policy = np.zeros((self.matsize+1,self.matsize+1),dtype=np.int)
        self.discountVal = 0.9