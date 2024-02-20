import matplotlib.pyplot as plt
import numpy as np
from keras.datasets import fashion_mnist
import keras

from activation import *
from Feedforward_Neural_Network import *
from pre_process import *


fashion_mnist=keras.datasets.fashion_mnist
(trainX, trainy), (testX, testy) = fashion_mnist.load_data()

def train_GD(layer_architecture, X_train, Y_train, X_test, Y_test, epochs=10, learning_rate=0.01, activation='relu'):
    np.random.seed(1) 
    nn = Feedforward_NeuralNetwork(layer_architecture, activation)
    m = Y_train.shape[1]
    
    for epoch in range(epochs):
        HL, caches = nn.forward_propagation(X_train)
        loss = nn.cross_entropy(Y_train, HL)
        grads = nn.backpropagation(X_train, Y_train, caches)
        nn.update_parameters(grads, learning_rate)
        
        if epoch % 1 == 0:

            print("Epoch %i, Training loss: %f" % (epoch, loss))
    
            # Evaluate model on whole test data after each epoch
            predictions, _ = nn.forward_propagation(X_test)
            accuracy = np.mean(np.argmax(predictions, axis=0) == np.argmax(Y_test, axis=0))
            print(f"Test accuracy : {accuracy}\n")


X_train, Y_train, X_test, Y_test = preprocess_data(trainX, trainy, testX, testy)
layer_architecture = [X_train.shape[0], 32, 64, 10]
train_GD(layer_architecture, X_train, Y_train, X_test, Y_test, epochs=3, activation='relu', learning_rate=0.1)
