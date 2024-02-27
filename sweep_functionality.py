import keras
import wandb
import numpy as np
from keras.datasets import fashion_mnist

from activation import *
from Feedforward_Neural_Network import *
from pre_process import *
from optimizers import *

wandb.login()

sweep_config = {
    'method': 'bayes', 
    'metric': {
        'name': 'val_accuracy', 'goal': 'maximize'},
    'parameters': {
        'epochs': {'values': [5, 10]},
        'num_layers': {'values': [3, 4, 5]},
        'hidden_size': {'values': [32, 64, 128]},
        'weight_decay': {'values': [0, 0.0005, 0.5]},
        'learning_rate': {'values': [1e-3, 1e-4]},
        'optimizer': {'values': ['sgd', 'momentum', 'nag', 'rmsprop', 'adam', 'nadam']},
        'batch_size': {'values': [16, 32, 64]},
        'weight_init': {'values': ['random', 'Xavier Normal']},
        'activation': {'values': ['sigmoid', 'tanh', 'relu']},
    }
}

def fit(layer_architecture, X_train, Y_train, X_val, Y_val, X_test, Y_test, epochs=100, activation='relu', loss = 'cross_entropy', optimizer='Nadam', weight_ini='Xavier Normal', learning_rate=0.001, beta=0.5, beta1=0.9, beta2=0.999, batch=16, weight_decay=0.0, epsilon=1e-6, project="cs23d014_assignment_1"):
    optimizer = optimizer.lower()

    if weight_ini == 'He':
        weight_ini = 'He Normal'
    elif weight_ini == 'Xavier':
        weight_ini = 'Xavier Normal'
        
    if optimizer == 'sgd':
        SGD(layer_architecture, X_train, Y_train, X_val, Y_val, X_test, Y_test, epochs=epochs, activation = activation,loss=loss, weight_ini = weight_ini, learning_rate=learning_rate, batch=batch, weight_decay=weight_decay, project="cs23d014_assignment_1")
    elif optimizer == 'momentum':
        MGD(layer_architecture, X_train, Y_train, X_val, Y_val, X_test, Y_test, epochs=epochs, activation=activation, loss=loss, weight_ini = weight_ini, learning_rate=learning_rate, beta=beta, batch=batch, weight_decay=weight_decay, project="cs23d014_assignment_1")
    elif optimizer == 'nag':
        NAG(layer_architecture, X_train, Y_train, X_val, Y_val, X_test, Y_test, epochs=epochs, activation=activation, loss=loss, weight_ini = weight_ini, learning_rate=learning_rate, beta=beta, batch=batch, weight_decay=weight_decay, project="cs23d014_assignment_1")
    elif optimizer == 'rmsprop':
        rmsprop(layer_architecture, X_train, Y_train, X_val, Y_val, X_test, Y_test, epochs=epochs, activation=activation, loss=loss, weight_ini = weight_ini, learning_rate=learning_rate, beta=beta, batch=batch, epsilon=epsilon, weight_decay=weight_decay, project="cs23d014_assignment_1")
    elif optimizer == 'adam':
        Adam(layer_architecture, X_train, Y_train, X_val, Y_val, X_test, Y_test, epochs=epochs, activation=activation, loss=loss, weight_ini = weight_ini, learning_rate=learning_rate, beta1=beta1, beta2=beta2, batch=batch, epsilon=epsilon, weight_decay=weight_decay)
    elif optimizer == 'nadam':
        Nadam(layer_architecture, X_train, Y_train, X_val, Y_val, X_test, Y_test, epochs=epochs, activation=activation, loss=loss, weight_ini = weight_ini, learning_rate=learning_rate, beta1=beta1, beta2=beta2, batch=batch, epsilon=epsilon, weight_decay=weight_decay)
    else:
        print('Please select optimizer correctly...')

def train():
    with wandb.init(project="cs23d014_assignment_1") as run:
        config = wandb.config  # Access hyperparameters via wandb.config

        if config.dataset == 'fashion_mnist':
            fashion_mnist = keras.datasets.fashion_mnist
            (trainX, trainy), (testX, testy) = fashion_mnist.load_data()
        else:
            mnist = keras.datasets.mnist
            (trainX, trainy), (testX, testy) = mnist.load_data() 

        shuffle = np.random.permutation(trainX.shape[0])
        trainX, trainy = trainX[shuffle], trainy[shuffle]

        val_samples = int(len(trainy) * 0.1 / 10)
        validation_inx = np.zeros(len(trainy), dtype=bool)

        for i in np.unique(trainy):
            inx = np.where(trainy == i)[0]
            tot_indices = inx[:val_samples]
            validation_inx[tot_indices] = True

        valX, valy = trainX[validation_inx], trainy[validation_inx]
        trainX, trainy = trainX[~validation_inx], trainy[~validation_inx]

        X_train, Y_train = preprocess_data(trainX, trainy)
        X_test, Y_test = preprocess_data(testX, testy)
        X_val, Y_val = preprocess_data(valX, valy)

        # Setup the model based on wandb.config
        layer_architecture = [784] + [config.hidden_size] * config.num_layers + [10]

        # Fit the model. Ensure fit() function logs metrics to wandb using wandb.log()
        fit(layer_architecture, X_train, Y_train, X_val, Y_val, X_test, Y_test,
            epochs=config.epochs, activation=config.activation, loss=config.loss,
            optimizer=config.optimizer, weight_init=config.weight_init,
            learning_rate=config.learning_rate, batch_size=config.batch_size,
            weight_decay=config.weight_decay, epsilon=config.eps, project=config.wandb_project)
        

sweep_id = wandb.sweep(sweep=sweep_config, project='cs23d014_assignment_1')
wandb.agent(sweep_id, train, count=10)