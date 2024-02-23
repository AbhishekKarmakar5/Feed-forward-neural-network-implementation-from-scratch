import numpy as np

def identity(x):
    return x

def identity_derivative(x):
    return 1.0

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return x > 0

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1 - np.tanh(x)**2