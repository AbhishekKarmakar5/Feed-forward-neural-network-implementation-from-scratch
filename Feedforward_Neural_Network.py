import numpy as np
from activation import *

class Feedforward_NeuralNetwork:
    def __init__(self, layers, activation, weight_ini='Xavier'):
        self.layers = layers
        self.activation = self.find_activation_functions(activation)
        self.activation_derivative = self.find_activation_derivative(activation)
        self.parameters = self.initialize_parameters(weight_ini)

    def find_activation_functions(self, activation):
        if activation == 'relu':
            return relu
        elif activation == 'sigmoid':
            return sigmoid
        elif activation == 'tanh':
            return tanh
        else:
            raise ValueError("Unsupported activation function")
        
    def find_activation_derivative(self, activation):
        if activation == 'relu':
            return relu_derivative
        elif activation == 'sigmoid':
            return sigmoid_derivative
        elif activation == 'tanh':
            return tanh_derivative
        else:
            print("Select Relu, Sigmoid or Tanh only...")
    
    def initialize_parameters(self, weight_ini):
        parameters = {}
        for l in range(1, len(self.layers)):
            if weight_ini == 'He Normal':
                sigma = np.sqrt(2/self.layers[l-1])
                parameters['W' + str(l)] = np.random.randn(self.layers[l], self.layers[l-1]) * sigma
            elif weight_ini == 'He Uniform':
                limit = np.sqrt(6/self.layers[l-1])
                parameters['W' + str(l)] = np.random.uniform(-limit, limit, (self.layers[l], self.layers[l-1]))
            elif weight_ini == 'Xavier Normal':
                sigma = np.sqrt(1/self.layers[l-1])
                parameters['W' + str(l)] = np.random.randn(self.layers[l], self.layers[l-1]) * sigma
            elif weight_ini == 'Xavier Uniform':
                limit = np.sqrt(6/(self.layers[l] + self.layers[l-1]))
                parameters['W' + str(l)] = np.random.uniform(-limit, limit, (self.layers[l], self.layers[l-1]))
            else: 
                parameters['W' + str(l)] = np.random.randn(self.layers[l], self.layers[l-1]) * 0.01
            
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
        previous_store = {}
        H = X
        L = len(self.parameters) // 2
        
        for l in range(1, L):
            H_prev = H
            A = np.dot(self.parameters['W' + str(l)], H_prev) + self.parameters['b' + str(l)]
            H = self.activation(A) 
            previous_store['A' + str(l)] = A
            previous_store['H' + str(l)] = H
        
        AL = np.dot(self.parameters['W' + str(L)], H) + self.parameters['b' + str(L)]
        HL = self.softmax(AL)
        previous_store['A' + str(L)] = AL
        previous_store['H' + str(L)] = HL
        return HL, previous_store
    
    def backpropagation(self, X, Y, previous_store):
        grads = {}
        L = len(self.parameters) // 2 # Number of layers
        m = X.shape[1]
        Y = Y.reshape(previous_store['H' + str(L)].shape) # Re-aranges it to the same shape as that of o/p layer.

        # Initializing backpropagation and Output layer gradient
        dAL = previous_store['H' + str(L)] - Y
        grads["delta_W" + str(L)] = 1./m * np.dot(dAL, previous_store['H' + str(L-1)].T)
        grads["delta_b" + str(L)] = 1./m * np.sum(dAL, axis=1, keepdims=True)

        for l in reversed(range(1, L)):
            dH = np.dot(self.parameters["W" + str(l+1)].T, dAL) # dH_prev
            dA = self.activation_derivative(previous_store['A' + str(l)]) * dH # Element wise multiplication between 2 vectors
            if l > 1:
                grads["delta_W" + str(l)] = 1./m * np.dot(dA, previous_store['H' + str(l-1)].T)
            else: # For the first hidden layer, use X 
                grads["delta_W" + str(l)] = 1./m * np.dot(dA, X.T)
            grads["delta_b" + str(l)] = 1./m * np.sum(dA, axis=1, keepdims=True)
            dAL = dA  # For the next iteration. Prepare dAL for next layer (if not the first layer)

        return grads
    
    def update_parameters(self, grads, learning_rate, weight_decay, m):
        L = len(self.parameters) // 2
        for l in range(L):
            self.parameters["W" + str(l+1)] -= learning_rate * (grads["delta_W" + str(l+1)] + (weight_decay / m) * self.parameters["W" + str(l+1)])
            self.parameters["b" + str(l+1)] -= learning_rate * grads["delta_b" + str(l+1)]

    def update_parameters_with_momentum_or_NAG(self, grads, learning_rate, beta, u_w_b, weight_decay, m):
        L = len(self.parameters) // 2
        
        for l in range(1, L+1):
            u_w_b["delta_W" + str(l)] = beta * u_w_b["delta_W" + str(l)] + learning_rate * grads["delta_W" + str(l)]
            u_w_b["delta_b" + str(l)] = beta * u_w_b["delta_b" + str(l)] + learning_rate * grads["delta_b" + str(l)]
            
            self.parameters["W" + str(l)] -= u_w_b["delta_W" + str(l)] + (weight_decay / m) * self.parameters["W" + str(l)]
            self.parameters["b" + str(l)] -= u_w_b["delta_b" + str(l)]
        
        return u_w_b

    def update_parameters_for_RMSprop(self, grads, learning_rate, beta, v_w_and_b, epsilon, weight_decay, m):
        L = len(self.parameters) // 2
        
        for l in range(1, L+1):
            # Compute intermediate values for gradients
            v_w_and_b['delta_W' + str(l)] = beta * v_w_and_b['delta_W' + str(l)] + (1 - beta) * np.square(grads['delta_W' + str(l)])
            v_w_and_b['delta_b' + str(l)] = beta * v_w_and_b['delta_b' + str(l)] + (1 - beta) * np.square(grads['delta_b' + str(l)])
            
            self.parameters['W' + str(l)] -= (learning_rate * grads['delta_W' + str(l)] / (np.sqrt(v_w_and_b['delta_W' + str(l)]) + epsilon)) + (learning_rate * (weight_decay / m) * self.parameters['W' + str(l)])
            self.parameters['b' + str(l)] -= learning_rate * grads['delta_b' + str(l)] / (np.sqrt(v_w_and_b['delta_b' + str(l)]) + epsilon)

        return v_w_and_b

    def update_parameters_for_Adam(self, learning_rate, m_w_and_b_hat_delta_W, v_w_and_b_hat_delta_W, m_w_and_b_hat_delta_b, v_w_and_b_hat_delta_b, l, epsilon, weight_decay, m):
        self.parameters['W' + str(l)] -= (learning_rate * m_w_and_b_hat_delta_W / (np.sqrt(v_w_and_b_hat_delta_W) + epsilon)) + (learning_rate * (weight_decay / m) * self.parameters['W' + str(l)])
        self.parameters['b' + str(l)] -= learning_rate * m_w_and_b_hat_delta_b / (np.sqrt(v_w_and_b_hat_delta_b) + epsilon)


    def update_parameters_for_Nadam(self, m_w_and_b,v_w_and_b, beta1, beta2, learning_rate, epoch, grads, epsilon, weight_decay, m):
        L = len(self.parameters) // 2
        for l in range(1, L+1):
            
            # computer intermediate values
            m_w_and_b_delta_W = beta1 * m_w_and_b['delta_W' + str(l)] + (1 - beta1) * grads['delta_W' + str(l)]
            m_w_and_b_delta_b = beta1 * m_w_and_b['delta_b' + str(l)] + (1 - beta1) * grads['delta_b' + str(l)]

            v_w_and_b_delta_W = beta2 * v_w_and_b['delta_W' + str(l)] + (1 - beta2) * np.square(grads['delta_W' + str(l)])
            v_w_and_b_delta_b = beta2 * v_w_and_b['delta_b' + str(l)] + (1 - beta2) * np.square(grads['delta_b' + str(l)])

            # Correct the bias
            m_w_and_b_hat_delta_W = m_w_and_b_delta_W / (1 - beta1 ** (epoch + 1))
            m_w_and_b_hat_delta_b = m_w_and_b_delta_b / (1 - beta1 ** (epoch + 1))
            v_w_and_b_hat_delta_W = v_w_and_b_delta_W / (1 - beta2 ** (epoch + 1))
            v_w_and_b_hat_delta_b = v_w_and_b_delta_b / (1 - beta2 ** (epoch + 1))

            self.parameters['W' + str(l)] -= learning_rate * ((beta1 * m_w_and_b_hat_delta_W + ((1 - beta1) * grads['delta_W' + str(l)]) / (1 - beta1 ** (epoch + 1))) / (np.sqrt(v_w_and_b_hat_delta_W) + epsilon) + (weight_decay / m) * self.parameters['W' + str(l)])
            self.parameters['b' + str(l)] -= learning_rate * (m_w_and_b_hat_delta_b + ((1 - beta1) * grads['delta_b' + str(l)]) / (1 - beta1 ** (epoch + 1))) / (np.sqrt(v_w_and_b_hat_delta_b) + epsilon)

            # Update moving averages
            m_w_and_b['delta_W' + str(l)] = m_w_and_b_delta_W
            m_w_and_b['delta_b' + str(l)] = m_w_and_b_delta_b
            v_w_and_b['delta_W' + str(l)] = v_w_and_b_delta_W
            v_w_and_b['delta_b' + str(l)] = v_w_and_b_delta_b

        return m_w_and_b, v_w_and_b
    