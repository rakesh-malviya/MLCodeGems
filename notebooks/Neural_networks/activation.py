import numpy as np
import math

class Sigmoid:
    def __init__(self):
        def sig(x):
            return 1 / (1 + math.exp(-x))

        def sig_grad(x):
            return sig(x)*sig(-x)

        self.func_vec = np.vectorize(sig)
        self.func_vec_grad = np.vectorize(sig_grad)


    def eval(self,z):
        return self.func_vec(z)

    def eval_grad(self,z):
        return self.func_vec_grad(z)


#a = np.array([1,2])
#sigmoid = Sigmoid()
