import wandb
import argparse

import numpy as np
from keras.datasets import fashion_mnist
import keras

from activation import *
from Feedforward_Neural_Network import *
from pre_process import *
from optimizers import *

def fit(layer_architecture, X_train, Y_train, X_val, Y_val, X_test, Y_test, epochs=100, activation='relu', loss = 'cross_entropy', optimizer='Nadam', weight_ini='Xavier Normal', learning_rate=0.001, 
        beta=0.5, beta1=0.9, beta2=0.999, batch_size=16, weight_decay=0.0, epsilon=1e-6, project="cs23d014_assignment_1", dataset='fashion_mnist'):
    optimizer = optimizer.lower()

    if weight_ini == 'He':
        weight_ini = 'He Normal'
    elif weight_ini == 'Xavier':
        weight_ini = 'Xavier Normal'

    to_run = 'd_CHECK'+str(dataset)+'_ep_'+str(epochs)+'_a_'+str(activation)+'_ls_'+str(loss)+'_bs_'+str(batch_size)+'_op_'+str(optimizer)+'_lr_'+str(learning_rate)+'_nhl_'+str(len(layer_architecture)-2)+'_sz_'+str(layer_architecture[1])+'_w_i_'+str(weight_ini)+'_w_d_'+str(weight_decay)
    wandb.init(project=project, name=to_run)
        
    if optimizer == 'sgd':
        SGD(layer_architecture, X_train, Y_train, X_val, Y_val, X_test, Y_test, epochs=epochs, activation = activation,loss=loss, weight_ini = weight_ini, learning_rate=learning_rate, batch_size=batch_size, 
            weight_decay=weight_decay, project="cs23d014_assignment_1", dataset=args.dataset)
    elif optimizer == 'momentum':
        MGD(layer_architecture, X_train, Y_train, X_val, Y_val, X_test, Y_test, epochs=epochs, activation=activation, loss=loss, weight_ini = weight_ini, learning_rate=learning_rate, beta=beta, batch_size=batch_size, 
            weight_decay=weight_decay, project="cs23d014_assignment_1", dataset=args.dataset)
    elif optimizer == 'nag':
        NAG(layer_architecture, X_train, Y_train, X_val, Y_val, X_test, Y_test, epochs=epochs, activation=activation, loss=loss, weight_ini = weight_ini, learning_rate=learning_rate, beta=beta, batch_size=batch_size, 
            weight_decay=weight_decay, project="cs23d014_assignment_1", dataset=args.dataset)
    elif optimizer == 'rmsprop':
        rmsprop(layer_architecture, X_train, Y_train, X_val, Y_val, X_test, Y_test, epochs=epochs, activation=activation, loss=loss, weight_ini = weight_ini, learning_rate=learning_rate, beta=beta, batch_size=batch_size, 
                epsilon=epsilon, weight_decay=weight_decay, project="cs23d014_assignment_1", dataset=args.dataset)
    elif optimizer == 'adam':
        Adam(layer_architecture, X_train, Y_train, X_val, Y_val, X_test, Y_test, epochs=epochs, activation=activation, loss=loss, weight_ini = weight_ini, learning_rate=learning_rate, beta1=beta1, beta2=beta2, batch_size=batch_size, 
             epsilon=epsilon, weight_decay=weight_decay, project="cs23d014_assignment_1", dataset=args.dataset)
    elif optimizer == 'nadam':
        Nadam(layer_architecture, X_train, Y_train, X_val, Y_val, X_test, Y_test, epochs=epochs, activation=activation, loss=loss, weight_ini = weight_ini, learning_rate=learning_rate, beta1=beta1, beta2=beta2, batch_size=batch_size, 
              epsilon=epsilon, weight_decay=weight_decay, project="cs23d014_assignment_1", dataset=args.dataset)
    else:
        print('Please select optimizer correctly...')

def train_arguments(args):
    wandb.login()

    if args.dataset == 'fashion_mnist':
        fashion_mnist=keras.datasets.fashion_mnist
        (trainX, trainy), (testX, testy) = fashion_mnist.load_data()
    else:
        mnist=keras.datasets.mnist
        (trainX, trainy), (testX, testy) = mnist.load_data()

    """
    The data has been shuffled so that while creating the validation dataset, random images from all classes can be picked.
    """
    shuffle = np.random.permutation(trainX.shape[0])
    trainX = trainX[shuffle]
    trainy = trainy[shuffle]

    val_samples = int(len(trainy)*0.1/10) # 10% data is kept for validation. Since there are 10 classes, therefore 'val_samples' are the no. of images per class shall be present in valX and valy
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

    
    ####################################################################################################
    ######                              Define Network Architecture                                 ####
    ######                      Change the layer architecture as per your requirement               ####
    ######                      Change the no. of hidden layers and the no. of neurons              ####
    ####################################################################################################
    """
    Length of 'layer_architecture' represents no. of network layers and the content inside the list represents the no. of neurons.
    To change the no. of neurons and no. of hidden layers dynamically, UN-COMMENT the line and COMMENT the line below .
    """
    # layer_architecture = [X_train.shape[0], 128, 64, 32, 32, 10] # <------------------------------------------Uncomment this line and comment the line below
    layer_architecture = [784] + [args.hidden_size]*args.num_layers + [10]

    fit(layer_architecture, X_train, Y_train, X_val, Y_val, X_test, Y_test, epochs=args.epochs, activation=args.activation, loss = args.loss, optimizer=args.optimizer, weight_ini = args.weight_init,
        learning_rate=args.learning_rate, batch_size=args.batch_size, weight_decay=args.weight_decay, epsilon=args.eps, project=args.wandb_project, dataset=args.dataset)


parser = argparse.ArgumentParser()
parser.add_argument('-wp', '--wandb_project', type=str, default='cs23d014_assignment_1', help='Choose the project name')
parser.add_argument('-we', '--wandb_entity', type=str, default='cs23d014', help='Choose the project entity')
parser.add_argument('-d', '--dataset', type=str, default='fashion_mnist', choices=['mnist', 'fashion_mnist'], help='The two choices are - mnist and fashion_mnist.')
parser.add_argument('-e', '--epochs', type=int, default=30,help='Number of epochs to train the model.')
parser.add_argument('-b', '--batch_size', type=int, default=256,help='Batch size required to train the model.')
parser.add_argument('-l', '--loss', type=str, default='cross_entropy', choices=['mean_squared_error', 'cross_entropy'],help='The two loss fucntion choices are - mean_squared_error and cross_entropy')
parser.add_argument('-o', '--optimizer', type=str, default='nadam', choices=['sgd', 'momentum', 'nag', 'rmsprop', 'adam', 'nadam'],help='Selection of optimizers.')
parser.add_argument('-lr', '--learning_rate', type=float, default=0.001,help='eta selection for suitable weights updates')
parser.add_argument('-m', '--momentum', type=float, default=0.5,help='momentum to speed of the process')
parser.add_argument('--beta', type=float, default=0.5, help='Beta used by rmsprop optimizer.')
parser.add_argument('--beta1', type=float, default=0.9,help='Beta1 used by adam and nadam optimizers')
parser.add_argument('--beta2', type=float, default=0.999,help='Beta2 used by adam and nadam optimizers.')
parser.add_argument('--eps', '--epsilon', type=float, default=0.0001,help='Epsilon used by optimizers.')
parser.add_argument('-w_d', '--weight_decay', type=float, default=0.0005,help='Weight decay for optimizers')
parser.add_argument('-w_i', '--weight_init', type=str, default='He Normal', choices=['He Normal', 'He Uniform', 'Xavier Normal', 'Xavier Uniform', 'random'],help='Weight initialization.')
parser.add_argument('-nhl', '--num_layers', type=int, default=4,help='No. of hidden layers in the feedforward neural network.')
parser.add_argument('-sz', '--hidden_size', type=int, default=128, help='No. of neurons in each hidden layer of the feedforward neural network.')
parser.add_argument('-a', '--activation', type=str, default='relu', choices=['identity', 'sigmoid', 'tanh', 'relu'])
args = parser.parse_args()
train_arguments(args) # python train.py --dataset mnist --epochs 100 -nhl 3 -sz 64