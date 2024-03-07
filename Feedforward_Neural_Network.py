import numpy as np
from activation import *

class Feedforward_NeuralNetwork:
    def __init__(self, layers, activation, weight_ini='Xavier', loss='cross_entopy'):
        self.layers = layers
        self.loss = loss
        self.activation = self.find_activation_functions(activation)
        self.activation_derivative = self.find_activation_derivative(activation)
        self.parameters = self.initialize_parameters(weight_ini)

    def find_activation_functions(self, activation):
        """
        Matches the activation function and then assigns that particular activation function to the layer
        """
        activation = activation.lower()
        if activation == 'identity':
            return identity
        elif activation == 'relu':
            return relu
        elif activation == 'sigmoid':
            return sigmoid
        elif activation == 'tanh':
            return tanh
        else:
            raise ValueError("Unsupported activation function")
        
    def find_activation_derivative(self, activation):
        """
        Matches the activation function and then returns its derivative.
        """
        activation = activation.lower()
        if activation == 'identity':
            return identity_derivative
        elif activation == 'relu':
            return relu_derivative
        elif activation == 'sigmoid':
            return sigmoid_derivative
        elif activation == 'tanh':
            return tanh_derivative
        else:
            print("Select Relu, Sigmoid or Tanh only...")
    
    def initialize_parameters(self, weight_ini):
        """
        The weights and bias of the respective layers are initialized.
        For weight inilialization He Normal, He Uniform, Xavier Normal, Xavier Uniform and Random has been used.
        """
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
                parameters['W' + str(l)] = np.random.randn(self.layers[l], self.layers[l-1]) * 0.001
            
            parameters['b' + str(l)] = np.zeros((self.layers[l], 1))
        
        return parameters

    
    def softmax(self, x):
        """
        The softmax function is used in the final output layer.
        """
        x_max = np.max(x, axis=0, keepdims=True)
        exponential_x = np.exp(x - x_max)
        sum_exponential_x = exponential_x.sum(axis=0, keepdims=True)
        softmax_x = exponential_x / sum_exponential_x
        return softmax_x
    
    def cross_entropy(self, Y, Y_hat):
        """
        Categorical cross entropy function as a loss function has been considered for y_pred and y_actual.
        """
        m = Y.shape[1]
        loss = -np.sum(Y * np.log(Y_hat + 1e-6))/m
        return loss
    
    def forward_propagation(self, X):
        """
        'H' acts as an i/p to the activation function and 'A' is the o/p after the activation.
        'previous_store' stores all the layer o/p so that we don't have to computer twice in case of backpropagation.
        """
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
        """
        Performs a backpropagation on all the layers based on the derivative of loss wrt the weight and bias on that particular neuron.
        """
        grads = {} # stores the gradient of all the layers
        L = len(self.parameters) // 2 # Number of layers
        m = X.shape[1]
        Y = Y.reshape(previous_store['H' + str(L)].shape) # Re-aranges it to the same shape as that of o/p layer.

        if self.loss == 'cross_entropy':
            dAL = previous_store['H' + str(L)] - Y
        elif self.loss == 'mean_squared_error':
            dAL = (previous_store['H' + str(L)] - Y) * self.activation_derivative(previous_store['A' + str(L)])
        else:
            print("Wrong error fn.")

        # Init. backpropagation and gradients at O/P
        grads["delta_W" + str(L)] = 1./m * np.dot(dAL, previous_store['H' + str(L-1)].T)
        grads["delta_b" + str(L)] = 1./m * np.sum(dAL, axis=1, keepdims=True)

        # Calculating the gradients of o/p layers first, then last hidden layer, this keeps on going until the first layer
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
        """
        The weights and bias are updated based on the SGD method.
        """
        L = len(self.parameters) // 2
        for l in range(L):
            self.parameters["W" + str(l+1)] -= learning_rate * (grads["delta_W" + str(l+1)] + (weight_decay / m) * self.parameters["W" + str(l+1)])
            self.parameters["b" + str(l+1)] -= learning_rate * grads["delta_b" + str(l+1)]

    def update_parameters_with_momentum_or_NAG(self, grads, learning_rate, beta, u_w_b, weight_decay, m):
        """
        The weights and bias are updated based on the Momentum and NAG method.
        """
        L = len(self.parameters) // 2
        
        for l in range(1, L+1):
            u_w_b["delta_W" + str(l)] = beta * u_w_b["delta_W" + str(l)] + learning_rate * grads["delta_W" + str(l)]
            u_w_b["delta_b" + str(l)] = beta * u_w_b["delta_b" + str(l)] + learning_rate * grads["delta_b" + str(l)]
            
            self.parameters["W" + str(l)] -= u_w_b["delta_W" + str(l)] + (weight_decay / m) * self.parameters["W" + str(l)]
            self.parameters["b" + str(l)] -= u_w_b["delta_b" + str(l)]
        
        return u_w_b

    def update_parameters_for_RMSprop(self, grads, learning_rate, beta, v_w_and_b, epsilon, weight_decay, m):
        """
        The weights and bias are updated based on the method of RMSprop.
        """
        L = len(self.parameters) // 2
        
        for l in range(1, L+1):
            # Compute intermediate values for gradients
            v_w_and_b['delta_W' + str(l)] = beta * v_w_and_b['delta_W' + str(l)] + (1 - beta) * np.square(grads['delta_W' + str(l)])
            v_w_and_b['delta_b' + str(l)] = beta * v_w_and_b['delta_b' + str(l)] + (1 - beta) * np.square(grads['delta_b' + str(l)])
            
            self.parameters['W' + str(l)] -= (learning_rate * grads['delta_W' + str(l)] / (np.sqrt(v_w_and_b['delta_W' + str(l)]) + epsilon)) + (learning_rate * (weight_decay / m) * self.parameters['W' + str(l)])
            self.parameters['b' + str(l)] -= learning_rate * grads['delta_b' + str(l)] / (np.sqrt(v_w_and_b['delta_b' + str(l)]) + epsilon)

        return v_w_and_b

    def update_parameters_for_Adam(self, learning_rate, m_w_and_b_hat_delta_W, v_w_and_b_hat_delta_W, m_w_and_b_hat_delta_b, v_w_and_b_hat_delta_b, l, epsilon, weight_decay, m):
        """
        The weights and bias updation rule followed by Adam optimizer.
        """
        self.parameters['W' + str(l)] -= (learning_rate * m_w_and_b_hat_delta_W / (np.sqrt(v_w_and_b_hat_delta_W) + epsilon)) + (learning_rate * (weight_decay / m) * self.parameters['W' + str(l)])
        self.parameters['b' + str(l)] -= learning_rate * m_w_and_b_hat_delta_b / (np.sqrt(v_w_and_b_hat_delta_b) + epsilon)


    def update_parameters_for_Nadam(self, m_w_and_b,v_w_and_b, beta1, beta2, learning_rate, epoch, grads, epsilon, weight_decay, m):
        """
        The weights and bias updation rule followed by Nadam optimizer.
        """
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
    