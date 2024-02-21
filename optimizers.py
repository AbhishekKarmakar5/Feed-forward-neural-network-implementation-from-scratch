import numpy as np
from activation import *
from Feedforward_Neural_Network import *

def test_accuracy(y_pred, Y_test):
    print("Test accuracy :", np.mean(np.argmax(y_pred, axis=0) == np.argmax(Y_test, axis=0)))

def do_SGD(layer_architecture, X_train, Y_train, X_test, Y_test, epochs=10, activation='relu', weight_ini = 'He', learning_rate=0.001, batch=1):
    nn = Feedforward_NeuralNetwork(layer_architecture, activation, weight_ini)
    
    m = X_train.shape[1]  # Number of training examples
    for epoch in range(epochs):
        epoch_loss = 0
        for i in range(0, m, batch):
            X = X_train[:, i:i+batch]
            Y = Y_train[:, i:i+batch]
            
            HL, caches = nn.forward_propagation(X)
            mini_batch_loss = nn.cross_entropy(Y, HL)
            epoch_loss += mini_batch_loss # Aggregate loss
            
            grads = nn.backpropagation(X, Y, caches)
            nn.update_parameters(grads, learning_rate)
        
        epoch_loss /= m # Average loss over all mini-batches
        
        print("Epoch ",epoch," Training loss: ", epoch_loss)
        y_pred, _ = nn.forward_propagation(X_test)
        test_accuracy(y_pred, Y_test)


def do_MGD(layer_architecture, X_train, Y_train, X_test, Y_test, epochs=10, activation='tanh',  weight_ini = 'He', learning_rate=0.001, beta=0.9, batch=1):
    nn = Feedforward_NeuralNetwork(layer_architecture, activation, weight_ini)
    m = X_train.shape[1]

    u_w_b = {} # history vectors
    for l in range(1, len(nn.layers)):
        u_w_b['delta_W' + str(l)] = np.zeros_like(nn.parameters['W' + str(l)])
        u_w_b['delta_b' + str(l)] = np.zeros_like(nn.parameters['b' + str(l)])
    
    for epoch in range(epochs):
        epoch_loss = 0
        for i in range(0, m, batch):
            X = X_train[:, i:i+batch]
            Y = Y_train[:, i:i+batch]
            
            HL, caches = nn.forward_propagation(X)
            mini_batch_loss = nn.cross_entropy(Y, HL)
            epoch_loss += mini_batch_loss
            
            grads = nn.backpropagation(X, Y, caches)
            nn.update_parameters_with_momentum_or_NAG(grads, learning_rate, beta, u_w_b)
        
        epoch_loss /= m 

        print("Epoch ",epoch," Training loss: ", epoch_loss)
        y_pred, _ = nn.forward_propagation(X_test)
        test_accuracy(y_pred, Y_test)


def do_NAG(layer_architecture, X_train, Y_train, X_test, Y_test, epochs=3, activation='tanh',  weight_ini = 'He', learning_rate=0.1, beta=0.9, batch=1):
    nn = Feedforward_NeuralNetwork(layer_architecture, activation, weight_ini)
    m = X_train.shape[1]

    prev_v_wb = {}  # history
    for l in range(1, len(nn.layers)):
        prev_v_wb['delta_W' + str(l)] = np.zeros_like(nn.parameters['W' + str(l)])
        prev_v_wb['delta_b' + str(l)] = np.zeros_like(nn.parameters['b' + str(l)])

    for epoch in range(epochs):
        epoch_loss = 0
        for i in range(0, m, batch):
            X = X_train[:, i:i+batch]
            Y = Y_train[:, i:i+batch]

            # Look-ahead step
            v_w_and_v_b = {}
            for l in range(1, len(nn.layers)):
                v_w_and_v_b["W" + str(l)] = nn.parameters["W" + str(l)] - beta * prev_v_wb["delta_W" + str(l)]
                v_w_and_v_b["b" + str(l)] = nn.parameters["b" + str(l)] - beta * prev_v_wb["delta_b" + str(l)]
            
            # Use v_w_and_v_b parameters for forward propagation
            nn.parameters, original_parameters = v_w_and_v_b, nn.parameters
            HL, caches = nn.forward_propagation(X)
            nn.parameters = original_parameters  # Restore original parameters

            mini_batch_loss = nn.cross_entropy(Y, HL)
            epoch_loss += mini_batch_loss

            grads = nn.backpropagation(X, Y, caches)
            prev_v_wb = nn.update_parameters_with_momentum_or_NAG(grads, learning_rate, beta, prev_v_wb)

        epoch_loss /= m  # Average loss over all mini-batches

        print("Epoch ",epoch," Training loss: ", epoch_loss)
        y_pred, _ = nn.forward_propagation(X_test)
        test_accuracy(y_pred, Y_test)


def do_rmsprop(layer_architecture, X_train, Y_train, X_test, Y_test, epochs=3, activation='tanh', weight_ini = 'He', learning_rate=0.01, beta=0.9, batch=1):
    nn = Feedforward_NeuralNetwork(layer_architecture, activation, weight_ini)
    m = X_train.shape[1]

    v_w_and_b = {}  # squared gradients summation for RMSprop
    for l in range(1, len(nn.layers)):
        v_w_and_b['delta_W' + str(l)] = np.zeros_like(nn.parameters['W' + str(l)])
        v_w_and_b['delta_b' + str(l)] = np.zeros_like(nn.parameters['b' + str(l)])

    for epoch in range(epochs):
        epoch_loss = 0
        for i in range(0, m, batch):
            X = X_train[:, i:i+batch]
            Y = Y_train[:, i:i+batch]

            HL, caches = nn.forward_propagation(X)
            mini_batch_loss = nn.cross_entropy(Y, HL)
            epoch_loss += mini_batch_loss

            grads = nn.backpropagation(X, Y, caches)
            v_w_and_b = nn.update_parameters_for_RMSprop(grads, learning_rate, beta, v_w_and_b)

        epoch_loss /= m  # Average loss over all mini-batches

        print("Epoch ",epoch," Training loss: ", epoch_loss)
        y_pred, _ = nn.forward_propagation(X_test)
        test_accuracy(y_pred, Y_test)

def do_Adam(layer_architecture, X_train, Y_train, X_test, Y_test, epochs=3, activation='tanh',  weight_ini = 'He', learning_rate=0.001, beta1=0.9, beta2=0.999,batch=1):
    nn = Feedforward_NeuralNetwork(layer_architecture, activation, weight_ini)
    m = X_train.shape[1]

    m_w_and_b = {}  # Momentum conservation
    v_w_and_b = {}  # RMSprop - Exp. Wt. Avg.
    for l in range(1, len(nn.layers)):
        m_w_and_b['delta_W' + str(l)] = np.zeros_like(nn.parameters['W' + str(l)])
        m_w_and_b['delta_b' + str(l)] = np.zeros_like(nn.parameters['b' + str(l)])
        v_w_and_b['delta_W' + str(l)] = np.zeros_like(nn.parameters['W' + str(l)])
        v_w_and_b['delta_b' + str(l)] = np.zeros_like(nn.parameters['b' + str(l)])

    for epoch in range(epochs):
        epoch_loss = 0
        for i in range(0, m, batch):
            X = X_train[:, i:i+batch]
            Y = Y_train[:, i:i+batch]

            HL, caches = nn.forward_propagation(X)
            mini_batch_loss = nn.cross_entropy(Y, HL)
            epoch_loss += mini_batch_loss

            grads = nn.backpropagation(X, Y, caches)

            for l in range(1, len(nn.layers)):
                m_w_and_b['delta_W' + str(l)] = beta1 * m_w_and_b['delta_W' + str(l)] + (1 - beta1) * grads['delta_W' + str(l)]
                m_w_and_b['delta_b' + str(l)] = beta1 * m_w_and_b['delta_b' + str(l)] + (1 - beta1) * grads['delta_b' + str(l)]

                v_w_and_b['delta_W' + str(l)] = beta2 * v_w_and_b['delta_W' + str(l)] + (1 - beta2) * np.square(grads['delta_W' + str(l)])
                v_w_and_b['delta_b' + str(l)] = beta2 * v_w_and_b['delta_b' + str(l)] + (1 - beta2) * np.square(grads['delta_b' + str(l)])

                # Correct the bias in first moment
                m_w_and_b_hat_delta_W = m_w_and_b['delta_W' + str(l)] / (1 - beta1 ** (epoch + 1))
                m_w_and_b_hat_delta_b = m_w_and_b['delta_b' + str(l)] / (1 - beta1 ** (epoch + 1))

                # Correct the bias in second moment
                v_w_and_b_hat_delta_W = v_w_and_b['delta_W' + str(l)] / (1 - beta2 ** (epoch + 1))
                v_w_and_b_hat_delta_b = v_w_and_b['delta_b' + str(l)] / (1 - beta2 ** (epoch + 1))

                nn.update_parameters_for_Adam(learning_rate, m_w_and_b_hat_delta_W, v_w_and_b_hat_delta_W, m_w_and_b_hat_delta_b, v_w_and_b_hat_delta_b, l)

        epoch_loss /= m 

        print("Epoch ",epoch," Training loss: ", epoch_loss)
        y_pred, _ = nn.forward_propagation(X_test)
        test_accuracy(y_pred, Y_test)


def do_Nadam(layer_architecture, X_train, Y_train, X_test, Y_test, epochs=3, activation='tanh',  weight_ini = 'He', learning_rate=0.01, beta1=0.9, beta2=0.999, batch=1):
    nn = Feedforward_NeuralNetwork(layer_architecture, activation, weight_ini)
    m = X_train.shape[1]

    m_w_and_b = {}  # Momentum
    v_w_and_b = {}  # Velocity
    for l in range(1, len(nn.layers)):
        m_w_and_b['delta_W' + str(l)] = np.zeros_like(nn.parameters['W' + str(l)])
        m_w_and_b['delta_b' + str(l)] = np.zeros_like(nn.parameters['b' + str(l)])
        v_w_and_b['delta_W' + str(l)] = np.zeros_like(nn.parameters['W' + str(l)])
        v_w_and_b['delta_b' + str(l)] = np.zeros_like(nn.parameters['b' + str(l)])

    for epoch in range(epochs):
        epoch_loss = 0
        for i in range(0, m, batch):
            X = X_train[:, i:i+batch]
            Y = Y_train[:, i:i+batch]

            HL, caches = nn.forward_propagation(X)
            mini_batch_loss = nn.cross_entropy(Y, HL)
            epoch_loss += mini_batch_loss

            grads = nn.backpropagation(X, Y, caches)

            m_w_and_b, v_w_and_b = nn.update_parameters_for_Nadam(m_w_and_b, v_w_and_b, beta1, beta2, learning_rate, epoch, grads)

        epoch_loss /= m

        print("Epoch ",epoch," Training loss: ", epoch_loss)
        y_pred, _ = nn.forward_propagation(X_test)
        test_accuracy(y_pred, Y_test)