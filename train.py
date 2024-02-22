import matplotlib.pyplot as plt
import numpy as np
from keras.datasets import fashion_mnist
import keras

from activation import *
from Feedforward_Neural_Network import *
from pre_process import *
from optimizers import *

def fit(layer_architecture, X_train, Y_train, X_test, Y_test, epochs=100, activation='relu',  optimizer='Nadam', weight_ini='Xavier Normal', learning_rate=0.001, beta=0.5, beta1=0.9, beta2=0.999, batch=16):
    if optimizer == 'SGD':
        do_SGD(layer_architecture, X_train, Y_train, X_test, Y_test, epochs=epochs, activation = activation, weight_ini = weight_ini, learning_rate=learning_rate, batch=batch)
    elif optimizer == 'MGD':
        do_MGD(layer_architecture, X_train, Y_train, X_test, Y_test, epochs=epochs, activation=activation, weight_ini = weight_ini, learning_rate=learning_rate, beta=beta, batch=batch)
    elif optimizer == 'NAG':
        do_NAG(layer_architecture, X_train, Y_train, X_test, Y_test, epochs=epochs, activation=activation, weight_ini = weight_ini, learning_rate=learning_rate, beta=beta, batch=batch)
    elif optimizer == ' RMSprop':
        do_rmsprop(layer_architecture, X_train, Y_train, X_test, Y_test, epochs=epochs, activation=activation, weight_ini = weight_ini, learning_rate=learning_rate, beta=beta, batch=batch)
    elif optimizer == 'Adam':
        do_Adam(layer_architecture, X_train, Y_train, X_test, Y_test, epochs=epochs, activation=activation, weight_ini = weight_ini, learning_rate=learning_rate, beta1=beta1, beta2=beta2, batch=batch)
    elif optimizer == 'Nadam':
        do_Nadam(layer_architecture, X_train, Y_train, X_test, Y_test, epochs=epochs, activation=activation, weight_ini = weight_ini, learning_rate=learning_rate, beta1=beta1, beta2=beta2, batch=batch)


fashion_mnist=keras.datasets.fashion_mnist
(trainX, trainy), (testX, testy) = fashion_mnist.load_data()

X_train, Y_train, X_test, Y_test = preprocess_data(trainX, trainy, testX, testy)
layer_architecture = [X_train.shape[0], 128, 64, 32, 32, 32, 10]

fit(layer_architecture, X_train, Y_train, X_test, Y_test, epochs=10, activation='tanh', optimizer='Adam', weight_ini = 'Xavier Uniform',learning_rate=0.0001, batch=32)