import matplotlib.pyplot as plt
import numpy as np
from keras.datasets import fashion_mnist
import keras

from activation import *
from Feedforward_Neural_Network import *
from pre_process import *
from optimizers import *

fashion_mnist=keras.datasets.fashion_mnist
(trainX, trainy), (testX, testy) = fashion_mnist.load_data()

X_train, Y_train, X_test, Y_test = preprocess_data(trainX, trainy, testX, testy)
layer_architecture = [X_train.shape[0], 32, 64, 10]
do_SGD(layer_architecture, X_train, Y_train, X_test, Y_test, epochs=3, activation='relu', learning_rate=0.001, batch=1)
do_MGD(layer_architecture, X_train, Y_train, X_test, Y_test, epochs=3, activation='tanh', learning_rate=0.01, beta=0.5, batch=1)
do_NAG(layer_architecture, X_train, Y_train, X_test, Y_test, epochs=30, activation='relu', learning_rate=0.01, beta=0.5, batch=1)


