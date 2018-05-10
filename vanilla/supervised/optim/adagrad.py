import numpy as np
from optimizer import Optimizer


class Adagrad(Optimizer):
    def __init__(self, params, lr=1e-1):
        super(Adagrad, self).__init__(params)
        """Adagrad algorithm"""
        self.lr = lr

    def __str__(self):
        return 'Adagrad algorithm. Learning rate: {}'.format(self.lr)
    
    def step(self):
        for _, param in self.params.items():
            param.m += param.d * param.d
            param.v += -(lr * param.d / np.sqrt(param.m + 1e-8))
