import matplotlib.pyplot as plt
import numpy as np
from keras.datasets import fashion_mnist
import keras

from activation import *
from Feedforward_Neural_Network import *
from pre_process import *
from optimizers import *

def fit(layer_architecture, X_train, Y_train, X_val, Y_val, X_test, Y_test, epochs=100, activation='relu',  optimizer='Nadam', weight_ini='Xavier Normal', learning_rate=0.001, beta=0.5, beta1=0.9, beta2=0.999, batch=16, weight_decay=0.0, epsilon=1e-6):
    if optimizer == 'sgd':
        SGD(layer_architecture, X_train, Y_train, X_val, Y_val, X_test, Y_test, epochs=epochs, activation = activation, weight_ini = weight_ini, learning_rate=learning_rate, batch=batch, weight_decay=weight_decay)
    elif optimizer == 'momentum':
        MGD(layer_architecture, X_train, Y_train, X_val, Y_val, X_test, Y_test, epochs=epochs, activation=activation, weight_ini = weight_ini, learning_rate=learning_rate, beta=beta, batch=batch, weight_decay=weight_decay)
    elif optimizer == 'nag':
        NAG(layer_architecture, X_train, Y_train, X_val, Y_val, X_test, Y_test, epochs=epochs, activation=activation, weight_ini = weight_ini, learning_rate=learning_rate, beta=beta, batch=batch, weight_decay=weight_decay)
    elif optimizer == 'rmsprop':
        rmsprop(layer_architecture, X_train, Y_train, X_val, Y_val, X_test, Y_test, epochs=epochs, activation=activation, weight_ini = weight_ini, learning_rate=learning_rate, beta=beta, batch=batch, epsilon=epsilon, weight_decay=weight_decay)
    elif optimizer == 'adam':
        Adam(layer_architecture, X_train, Y_train, X_val, Y_val, X_test, Y_test, epochs=epochs, activation=activation, weight_ini = weight_ini, learning_rate=learning_rate, beta1=beta1, beta2=beta2, batch=batch, epsilon=epsilon, weight_decay=weight_decay)
    elif optimizer == 'nadam':
        Nadam(layer_architecture, X_train, Y_train, X_val, Y_val, X_test, Y_test, epochs=epochs, activation=activation, weight_ini = weight_ini, learning_rate=learning_rate, beta1=beta1, beta2=beta2, batch=batch, epsilon=epsilon, weight_decay=weight_decay)
    else:
        print('Please selection optimizer correctly..')

fashion_mnist=keras.datasets.fashion_mnist
(trainX, trainy), (testX, testy) = fashion_mnist.load_data()

# Shuffled data to make train and val
shuffle = np.random.permutation(trainX.shape[0])
trainX = trainX[shuffle]
trainy = trainy[shuffle]

val_samples = int(len(trainy)*0.1/10)
validation_inx = np.zeros(len(trainy), dtype=bool)

for i in np.unique(trainy):
    inx = np.where(trainy == i)[0]
    tot_indices = inx[:val_samples]
    validation_inx[tot_indices] = True

valX = trainX[validation_inx]
valy = trainy[validation_inx]
trainX = trainX[~validation_inx]
trainy = trainy[~validation_inx]

X_train, Y_train = preprocess_data(trainX, trainy)
X_test, Y_test = preprocess_data(testX, testy)
X_val, Y_val = preprocess_data(valX, valy)

layer_architecture = [X_train.shape[0], 64, 10]
fit(layer_architecture, X_train, Y_train, X_val, Y_val, X_test, Y_test, epochs=3, activation='tanh', optimizer='nag', weight_ini = 'Xavier Normal',learning_rate=0.001, batch=1, weight_decay=0.0005, epsilon=1e-6)