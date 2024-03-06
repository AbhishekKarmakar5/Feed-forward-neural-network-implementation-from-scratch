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
    'name': 'Sweep_Fashion_Mnist_MSE_CE_Bayes_4',
    'method': 'bayes', 
    'metric': {'name': 'Validation accuracy ', 'goal': 'maximize'},
    'parameters': {
        'epochs': {'values': [5, 10, 15, 20, 25, 30]},
        'num_layers': {'values': [1, 2, 3, 4, 5]},
        'hidden_size': {'values': [16, 32, 64, 128, 256]},
        'weight_decay': {'values': [0, 0.001, 0.006, 0.0001, 0.0005, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]},
        'learning_rate': {'values': [0.01, 1e-2, 1e-3, 2e-3, 1e-4, 4e-4]},
        'optimizer': {'values': ['sgd', 'momentum', 'nag', 'rmsprop', 'adam', 'nadam']},
        'batch_size': {'values': [16, 32, 64, 128, 256]},
        'weight_ini': {'values': ['Xavier Uniform','Xavier Normal', 'He Uniform', 'He Normal']},
        'activation': {'values': ['relu', 'tanh', 'sigmoid']},  
        'dataset':{'values':['fashion_mnist']},
        'loss':{'values':['cross_entropy','mean_squared_error']},
        'eps':{'values':[0.0001, 1e-6]},
        'wandb_project':{'values':['cs23d014_assignment_1']}
    }
}
    
def fit(layer_architecture, X_train, Y_train, X_val, Y_val, X_test, Y_test, epochs=100, activation='relu', loss = 'cross_entropy', optimizer='Nadam', weight_ini='Xavier Normal', 
        learning_rate=0.001, beta=0.5, beta1=0.9, beta2=0.999, batch_size=16, weight_decay=0.0, epsilon=1e-6, project="cs23d014_assignment_1"):
    optimizer = optimizer.lower()


    if weight_ini == 'He':
        weight_ini = 'He Normal'
    elif weight_ini == 'Xavier':
        weight_ini = 'Xavier Normal'
        
    if optimizer == 'sgd':
        SGD(layer_architecture, X_train, Y_train, X_val, Y_val, X_test, Y_test, epochs=epochs, activation = activation,loss=loss, weight_ini = weight_ini, learning_rate=learning_rate, batch_size=batch_size, weight_decay=weight_decay, project="cs23d014_assignment_1")
    elif optimizer == 'momentum':
        MGD(layer_architecture, X_train, Y_train, X_val, Y_val, X_test, Y_test, epochs=epochs, activation=activation, loss=loss, weight_ini = weight_ini, learning_rate=learning_rate, beta=beta, batch_size=batch_size, weight_decay=weight_decay, project="cs23d014_assignment_1")
    elif optimizer == 'nag':
        NAG(layer_architecture, X_train, Y_train, X_val, Y_val, X_test, Y_test, epochs=epochs, activation=activation, loss=loss, weight_ini = weight_ini, learning_rate=learning_rate, beta=beta, batch_size=batch_size, weight_decay=weight_decay, project="cs23d014_assignment_1")
    elif optimizer == 'rmsprop':
        rmsprop(layer_architecture, X_train, Y_train, X_val, Y_val, X_test, Y_test, epochs=epochs, activation=activation, loss=loss, weight_ini = weight_ini, learning_rate=learning_rate, beta=beta, batch_size=batch_size, epsilon=epsilon, weight_decay=weight_decay, project="cs23d014_assignment_1")
    elif optimizer == 'adam':
        Adam(layer_architecture, X_train, Y_train, X_val, Y_val, X_test, Y_test, epochs=epochs, activation=activation, loss=loss, weight_ini = weight_ini, learning_rate=learning_rate, beta1=beta1, beta2=beta2, batch_size=batch_size, epsilon=epsilon, weight_decay=weight_decay)
    elif optimizer == 'nadam':
        Nadam(layer_architecture, X_train, Y_train, X_val, Y_val, X_test, Y_test, epochs=epochs, activation=activation, loss=loss, weight_ini = weight_ini, learning_rate=learning_rate, beta1=beta1, beta2=beta2, batch_size=batch_size, epsilon=epsilon, weight_decay=weight_decay)
    else:
        print('Please select optimizer correctly...')

def train():
    with wandb.init(project="cs23d014_assignment_1"):
        config = wandb.config  # Access hyperparameters via wandb.config
        wandb.run.name = 'd_'+str(config.dataset)+'_ep_'+str(config.epochs)+'_a_'+str(config.activation)+'_ls_'+str(config.loss)+'_bs_'+str(config.batch_size)+'_op_'+str(config.optimizer)+'_lr_'+str(config.learning_rate)+'_nhl_'+ str(config.num_layers)+'_sz_'+str(config.hidden_size)+'_w_i_'+config.weight_ini+'_w_d_'+str(config.weight_decay)

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

        layer_architecture = [784] + [config.hidden_size] * config.num_layers + [10]

        # Fit the model. Ensure fit() function logs metrics to wandb using wandb.log()
        fit(layer_architecture, X_train, Y_train, X_val, Y_val, X_test, Y_test,
            epochs=config.epochs, activation=config.activation, loss=config.loss,
            optimizer=config.optimizer, weight_ini=config.weight_ini,
            learning_rate=config.learning_rate, batch_size=config.batch_size,
            weight_decay=config.weight_decay, epsilon=config.eps, project=config.wandb_project)
        
        wandb.run.save()
        
sweep_id = wandb.sweep(sweep=sweep_config, project='cs23d014_assignment_1')
wandb.agent(sweep_id, train, count=100)
wandb.finish()