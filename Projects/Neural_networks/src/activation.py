import numpy as np
import math

class Sigmoid:
    def __init__(self):
        def sig(x):
            return 1 / (1 + math.exp(-x))

        def sig_grad(x):
            sigx = sig(x)
            return sigx*(1-sigx)

        self.func_vec = np.vectorize(sig)
        self.func_vec_grad = np.vectorize(sig_grad)


    def eval(self,z):
        return self.func_vec(z)

    def eval_grad(self,z):
        return self.func_vec_grad(z)

class Tanh:
    def __init__(self):
        def tanh(x):
            return math.tanh(x)

        def tanh_grad(x):
            tanhx = tanh(x)
            return (1- tanhx) * (1+tanhx)

        self.func_vec = np.vectorize(tanh)
        self.func_vec_grad = np.vectorize(tanh_grad)

    def eval(self, z):
        return self.func_vec(z)

    def eval_grad(self, z):
        return self.func_vec_grad(z)

#a = np.array([1,2])
#sigmoid = Sigmoid()
