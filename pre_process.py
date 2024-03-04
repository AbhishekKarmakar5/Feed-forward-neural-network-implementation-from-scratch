import numpy as np

def one_hot_encode(labels):
    return np.eye(10)[labels].T

def preprocess_data(inp_X, inp_y):
    # flattens then normalize. Finally one-hot encoding is performed for cross-entropy loss.
    inp_X = inp_X.reshape(inp_X.shape[0], -1).T / 255
    inp_y = one_hot_encode(inp_y)

    return inp_X, inp_y