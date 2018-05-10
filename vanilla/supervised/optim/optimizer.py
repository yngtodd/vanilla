import numpy as np


class Optimizer(object):
    def __init__(self, params):
        """base class for optimizers"""
        self.params = params

    def zero_grad(self):
        for _, param in self.params.items():
            param.d.fill(0)

    def step(self):
        raise NotImplementedError

