import numpy as np

def one_hot_encode(labels):
    return np.eye(10)[labels].T

def preprocess_data(trainX, trainy, testX, testy):
    X_train = trainX.reshape(trainX.shape[0], -1).T / 255.
    X_test = testX.reshape(testX.shape[0], -1).T / 255.
    
    Y_train = one_hot_encode(trainy)
    Y_test = one_hot_encode(testy)
    
    return X_train, Y_train, X_test, Y_test 