import numpy as np


def sigmoid(x):
    """Sigmoid activation"""
    return 1 / (1 + np.exp(-x))


def dsigmoid(x):
    """Derivative of sigmoid"""
    return x * (1 - x)


def tanh(x):
    """Hyperbolic tangent activation"""
    return np.tanh(x)


def dtanh(x):
    """Derivative of tanh"""
    return 1 - x * x
