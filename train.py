import matplotlib.pyplot as plt
import numpy as np
from keras.datasets import fashion_mnist
import keras

from activation import *
from Feedforward_Neural_Network import *
from pre_process import *
from optimizers import *

def fit(layer_architecture, X_train, Y_train, X_val, Y_val, X_test, Y_test, epochs=100, activation='relu', loss = 'cross_entropy', optimizer='Nadam', weight_ini='Xavier Normal', learning_rate=0.001, beta=0.5, beta1=0.9, beta2=0.999, batch=16, weight_decay=0.0, epsilon=1e-6):
    optimizer = optimizer.lower()

    if weight_ini == 'He':
        weight_ini = 'He Normal'
    elif weight_ini == 'Xavier':
        weight_ini = 'Xavier Normal'
        
    if optimizer == 'sgd':
        SGD(layer_architecture, X_train, Y_train, X_val, Y_val, X_test, Y_test, epochs=epochs, activation = activation,loss=loss, weight_ini = weight_ini, learning_rate=learning_rate, batch=batch, weight_decay=weight_decay)
    elif optimizer == 'momentum':
        MGD(layer_architecture, X_train, Y_train, X_val, Y_val, X_test, Y_test, epochs=epochs, activation=activation, loss=loss, weight_ini = weight_ini, learning_rate=learning_rate, beta=beta, batch=batch, weight_decay=weight_decay)
    elif optimizer == 'nag':
        NAG(layer_architecture, X_train, Y_train, X_val, Y_val, X_test, Y_test, epochs=epochs, activation=activation, loss=loss, weight_ini = weight_ini, learning_rate=learning_rate, beta=beta, batch=batch, weight_decay=weight_decay)
    elif optimizer == 'rmsprop':
        rmsprop(layer_architecture, X_train, Y_train, X_val, Y_val, X_test, Y_test, epochs=epochs, activation=activation, loss=loss, weight_ini = weight_ini, learning_rate=learning_rate, beta=beta, batch=batch, epsilon=epsilon, weight_decay=weight_decay)
    elif optimizer == 'adam':
        Adam(layer_architecture, X_train, Y_train, X_val, Y_val, X_test, Y_test, epochs=epochs, activation=activation, loss=loss, weight_ini = weight_ini, learning_rate=learning_rate, beta1=beta1, beta2=beta2, batch=batch, epsilon=epsilon, weight_decay=weight_decay)
    elif optimizer == 'nadam':
        Nadam(layer_architecture, X_train, Y_train, X_val, Y_val, X_test, Y_test, epochs=epochs, activation=activation, loss=loss, weight_ini = weight_ini, learning_rate=learning_rate, beta1=beta1, beta2=beta2, batch=batch, epsilon=epsilon, weight_decay=weight_decay)
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

layer_architecture = [X_train.shape[0], 64, 10] # cross_entropy # mean_squared_error
fit(layer_architecture, X_train, Y_train, X_val, Y_val, X_test, Y_test, epochs=15, activation='tanh', loss = 'cross_entropy', optimizer='nadam', weight_ini = 'Xavier Normal',learning_rate=0.0001, batch=256, weight_decay=0.0005, epsilon=1e-6)


# # -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# def train_arguments(args):
#     if args.dataset == 'mnist':
#         fashion_mnist=keras.datasets.fashion_mnist
#         (trainX, trainy), (testX, testy) = fashion_mnist.load_data()
#     else:
#         mnist=keras.datasets.mnist
#         (trainX, trainy), (testX, testy) = mnist.load_data()

#     # Shuffled data to make train and val
#     shuffle = np.random.permutation(trainX.shape[0])
#     trainX = trainX[shuffle]
#     trainy = trainy[shuffle]

#     val_samples = int(len(trainy)*0.1/10)
#     validation_inx = np.zeros(len(trainy), dtype=bool)

#     for i in np.unique(trainy):
#         inx = np.where(trainy == i)[0]
#         tot_indices = inx[:val_samples]
#         validation_inx[tot_indices] = True

#     valX = trainX[validation_inx]
#     valy = trainy[validation_inx]
#     trainX = trainX[~validation_inx]
#     trainy = trainy[~validation_inx]

#     X_train, Y_train = preprocess_data(trainX, trainy)
#     X_test, Y_test = preprocess_data(testX, testy)
#     X_val, Y_val = preprocess_data(valX, valy)


# # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# import argparse


# parser = argparse.ArgumentParser()
# parser.add_argument('-d', '--dataset', type=str, default='fashion_mnist', choices=['mnist', 'fashion_mnist'], help='The two choices are - mnist and fashion_mnist.')
# parser.add_argument('-e', '--epochs', type=int, default=15,help='Number of epochs to train the model.')
# parser.add_argument('-b', '--batch_size', type=int, default=32,help='Batch size required to train the model.')
# parser.add_argument('-l', '--loss', type=str, default='cross_entropy', choices=['mean_squared_error', 'cross_entropy'],help='The two loss fucntion choices are - mean_squared_error and cross_entropy')
# parser.add_argument('-o', '--optimizer', type=str, default='nadam', choices=['sgd', 'momentum', 'nag', 'rmsprop', 'adam', 'nadam'],help='Selection of optimizers.')
# parser.add_argument('-lr', '--learning_rate', type=float, default=0.003,help='eta selection for suitable weights updates')
# parser.add_argument('-m', '--momentum', type=float, default=0.5,help='momentum to speed of the process')
# parser.add_argument('--beta', type=float, default=0.5, help='Beta used by rmsprop optimizer.')
# parser.add_argument('--beta1', type=float, default=0.9,help='Beta1 used by adam and nadam optimizers')
# parser.add_argument('--beta2', type=float, default=0.999,help='Beta2 used by adam and nadam optimizers.')
# parser.add_argument('--eps', '--epsilon', type=float, default=0.0001,help='Epsilon used by optimizers.')
# parser.add_argument('-w_d', '--weight_decay', type=float, default=0.0,help='Weight decay for optimizers')
# parser.add_argument('-w_i', '--weight_init', type=str, default='random', choices=['He Normal', 'He Uniform', 'Xavier Normal', 'Xavier Uniform', 'random'],help='Weight initialization.')
# parser.add_argument('-nhl', '--num_layers', type=int, default=1,help='No. of hidden layers in the feedforward neural network.')
# parser.add_argument('-sz', '--hidden_size', type=int, default=32, help='No. of neurons in each hidden layer of the feedforward neural network.')
# parser.add_argument('-a', '--activation', type=str, default='sigmoid', choices=['identity', 'sigmoid', 'tanh', 'ReLU'])
# args = parser.parse_args()
# print(args) 
# print(args.dataset, args.epochs, args.batch_size)