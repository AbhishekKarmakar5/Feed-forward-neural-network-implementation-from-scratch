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
    # For positive values of x, the standard sigmoid has been used
    pos_x = x >= 0
    output_x = np.zeros_like(x, dtype=np.float64)
    z = np.exp(-x[pos_x])
    output_x[pos_x] = 1 / (1 + z)
    # For negative values of x, e^(x)/(e^(x) + 1) has been used.
    neg_x = ~pos_x
    z = np.exp(x[neg_x])
    output_x[neg_x] = z / (1 + z)
    
    return output_x

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1 - np.tanh(x)**2