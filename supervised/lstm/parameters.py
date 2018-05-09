import numpy as np


class Param:
    def __init__(self, name, value):
        """
        Basic parameter class.

        Parameters:
        ----------
        * `name`: [str]
            Name of the parameter.
        
        * `value`: [numpy array]
            Value for the parameter.

        * `d`: [numpy array]
            Derivative of the parameter values.
        
        * `m`: [numpy array]
            Momentum for AdaGrad.
        """
        self.name = name
        self.v = value
        self.d = np.zeros_like(value)
        self.m = np.zeros_like(value)



class Parameters:
    def __init__(self, z_size, H_size, n_classes, weight_sd=0.1):
        """
        Parameters of our LSTM.

        Parameters:
        ----------
        * `H_size`: [int, default=100]
            Hidden dimension size.

        * `z_size`: [int]
            Size of concatenate (H, X) vector. 

        * `W_*`: [numpy array]
            Weights of the net at layer *.

        * `b_*`: [numpy array]
            Biases of the net at layer *.

        * `n_classes`: [int]
            Number of classes to predict.

        *  `weight_sd`: [float, default=0.1]
            Standard deviation of weights for initizialization.
        """
        self.W_f = Param('W_f', np.random.randn(H_size, z_size) * weight_sd + 0.5)
        self.b_f = Param('b_f', np.zeros((H_size, 1)))
        self.W_i = Param('W_i', np.random.randn(H_size, z_size) * weight_sd + 0.5)
        self.b_i = Param('b_i', np.zeros((H_size, 1)))
        self.W_C = Param('W_C', np.random.randn(H_size, z_size) * weight_sd)
        self.b_C = Param('b_C', np.zeros((H_size, 1)))
        self.W_o = Param('W_o', np.random.randn(H_size, z_size) * weight_sd + 0.5)
        self.b_o = Param('b_o', np.zeros((H_size, 1)))
        self.W_v = Param('W_v', np.random.randn(n_classes, H_size) * weight_sd)
        self.b_v = Param('b_v', np.zeros((n_classes, 1)))

    def all(self):
        return [self.W_f, self.W_i, self.W_C, self.W_o, self.W_v,
                self.b_f, self.b_i, self.b_C, self.b_o, self.b_v]
