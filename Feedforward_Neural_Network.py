import numpy as np
from activation import *

class Feedforward_NeuralNetwork:
    def __init__(self, layers, activation):
        self.layers = layers
        self.activation, self.activation_derivative = self.set_activation_functions(activation)
        self.parameters = self.initialize_parameters(activation)

    def set_activation_functions(self, activation):
        if activation == 'relu':
            return relu, relu_derivative
        elif activation == 'sigmoid':
            return sigmoid, sigmoid_derivative
        elif activation == 'tanh':
            return tanh, tanh_derivative
        else:
            raise ValueError("Unsupported activation function")
    
    def initialize_parameters(self, activation):
        parameters = {}
        for l in range(1, len(self.layers)):
            if activation == 'relu':
                std_dev = np.sqrt(2. / self.layers[l-1]) # He Init
            else:
                std_dev = np.sqrt(1. / self.layers[l-1]) # Xavier Normal
                
            parameters['W' + str(l)] = np.random.randn(self.layers[l], self.layers[l-1]) * std_dev
            parameters['b' + str(l)] = np.zeros((self.layers[l], 1))
        return parameters
    
    def softmax(self, A):
        exponential_A = np.exp(A - np.max(A))
        return exponential_A / exponential_A.sum(axis=0, keepdims=True)
    
    def cross_entropy(self, Y, Y_hat):
        m = Y.shape[1]
        loss = -np.sum(Y * np.log(Y_hat + 1e-9))/m
        return loss
    
    def forward_propagation(self, X):
        caches = {}
        H = X
        L = len(self.parameters) // 2
        
        for l in range(1, L):
            H_prev = H
            A = np.dot(self.parameters['W' + str(l)], H_prev) + self.parameters['b' + str(l)]
            H = self.activation(A) 
            caches['A' + str(l)] = A
            caches['H' + str(l)] = H
        
        AL = np.dot(self.parameters['W' + str(L)], H) + self.parameters['b' + str(L)]
        HL = self.softmax(AL)
        caches['A' + str(L)] = AL
        caches['H' + str(L)] = HL
        return HL, caches
    
    def backpropagation(self, X, Y, caches):
        grads = {}
        L = len(self.parameters) // 2 # Number of layers
        m = X.shape[1]
        Y = Y.reshape(caches['H' + str(L)].shape) # Ensure same shape as output layer

        # Initializing backpropagation and Output layer gradient
        dAL = caches['H' + str(L)] - Y
        grads["dW" + str(L)] = 1./m * np.dot(dAL, caches['H' + str(L-1)].T)
        grads["db" + str(L)] = 1./m * np.sum(dAL, axis=1, keepdims=True)

        for l in reversed(range(1, L)):
            dH = np.dot(self.parameters["W" + str(l+1)].T, dAL) # dH_prev
            dA = self.activation_derivative(caches['A' + str(l)]) * dH # Element wise multiplication between 2 vectors
            if l > 1:
                grads["dW" + str(l)] = 1./m * np.dot(dA, caches['H' + str(l-1)].T)
            else: # For the first hidden layer, use X 
                grads["dW" + str(l)] = 1./m * np.dot(dA, X.T)
            grads["db" + str(l)] = 1./m * np.sum(dA, axis=1, keepdims=True)
            dAL = dA  # For the next iteration. Prepare dAL for next layer (if not the first layer)

        return grads
    
    def update_parameters(self, grads, learning_rate):
        L = len(self.parameters) // 2
        for l in range(L):
            self.parameters["W" + str(l+1)] -= learning_rate * grads["dW" + str(l+1)]
            self.parameters["b" + str(l+1)] -= learning_rate * grads["db" + str(l+1)]