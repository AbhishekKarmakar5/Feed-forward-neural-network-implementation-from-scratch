import numpy as np
from activation import *
from Feedforward_Neural_Network import *

def test_accuracy(y_pred, Y_test, acc_of='Validation'):
    print(acc_of, " accuracy :", np.mean(np.argmax(y_pred, axis=0) == np.argmax(Y_test, axis=0)))

def compute_accuracy(y_pred, Y):
    return np.mean(np.argmax(y_pred, axis=0) == np.argmax(Y, axis=0))

def compute_loss(Y, HL, nn):
    return nn.cross_entropy(Y, HL)

def SGD(layer_architecture, X_train, Y_train, X_val, Y_val, X_test, Y_test, epochs=10, activation='relu', weight_ini = 'He', learning_rate=0.001, batch=1, weight_decay=0.0):
    nn = Feedforward_NeuralNetwork(layer_architecture, activation, weight_ini)
    
    m = X_train.shape[1]  
    for epoch in range(epochs):
        epoch_loss = 0
        for i in range(0, m, batch):
            X = X_train[:, i:i+batch]
            Y = Y_train[:, i:i+batch]
            
            HL, previous_store = nn.forward_propagation(X)
            mini_batch_loss = nn.cross_entropy(Y, HL)

            l2_reg_loss = 0
            for l in range(1, len(layer_architecture)):
                l2_reg_loss += np.sum(np.square(nn.parameters['W' + str(l)]))
            l2_reg_loss = (weight_decay / (2 * m)) * l2_reg_loss
            epoch_loss += mini_batch_loss + l2_reg_loss
            
            grads = nn.backpropagation(X, Y, previous_store)
            nn.update_parameters(grads, learning_rate, weight_decay, batch)
        
        epoch_loss /= m

        # Training accuracy and loss
        y_pred_train, _ = nn.forward_propagation(X_train)
        train_accuracy = compute_accuracy(y_pred_train, Y_train)
        train_loss = compute_loss(Y_train, y_pred_train, nn) / m
        
        # Validation accuracy and loss
        y_pred_val, _ = nn.forward_propagation(X_val)
        val_accuracy = compute_accuracy(y_pred_val, Y_val)
        val_loss = compute_loss(Y_val, y_pred_val, nn) / X_val.shape[1] # Normalize 
        
        # Testing accuracy and loss
        y_pred_test, _ = nn.forward_propagation(X_test)
        test_accuracy = compute_accuracy(y_pred_test, Y_test)
        test_loss = compute_loss(Y_test, y_pred_test, nn) / X_test.shape[1]  # Normalize
        
        print("\nEpoch ",epoch, 'Epoch Loss : ', epoch_loss)
        print("Training loss: ", train_loss, " Training accuracy: ", train_accuracy)
        print("Validation loss: ", val_loss, " Validation accuracy: ", val_accuracy)
        print("Testing loss: ", test_loss, " Testing accuracy: ", test_accuracy)


def MGD(layer_architecture, X_train, Y_train, X_val, Y_val, X_test, Y_test, epochs=10, activation='tanh',  weight_ini = 'He', learning_rate=0.001, beta=0.9, batch=1, weight_decay=0.0):
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
            
            HL, previous_store = nn.forward_propagation(X)
            mini_batch_loss = nn.cross_entropy(Y, HL)

            l2_reg_loss = 0
            for l in range(1, len(layer_architecture)):
                l2_reg_loss += np.sum(np.square(nn.parameters['W' + str(l)]))
            l2_reg_loss = (weight_decay / (2 * m)) * l2_reg_loss
            epoch_loss += mini_batch_loss + l2_reg_loss
            
            grads = nn.backpropagation(X, Y, previous_store)
            nn.update_parameters_with_momentum_or_NAG(grads, learning_rate, beta, u_w_b, weight_decay, batch)
        
        epoch_loss /= m

        # Training accuracy and loss
        y_pred_train, _ = nn.forward_propagation(X_train)
        train_accuracy = compute_accuracy(y_pred_train, Y_train)
        train_loss = compute_loss(Y_train, y_pred_train, nn) / m
        
        # Validation accuracy and loss
        y_pred_val, _ = nn.forward_propagation(X_val)
        val_accuracy = compute_accuracy(y_pred_val, Y_val)
        val_loss = compute_loss(Y_val, y_pred_val, nn) / X_val.shape[1]  # Normalize 
        
        # Testing accuracy and loss
        y_pred_test, _ = nn.forward_propagation(X_test)
        test_accuracy = compute_accuracy(y_pred_test, Y_test)
        test_loss = compute_loss(Y_test, y_pred_test, nn) / X_test.shape[1] # Normalize
        
        print("\nEpoch ",epoch, 'Epoch Loss : ', epoch_loss)
        print("Training loss: ", train_loss, " Training accuracy: ", train_accuracy)
        print("Validation loss: ", val_loss, " Validation accuracy: ", val_accuracy)
        print("Testing loss: ", test_loss, " Testing accuracy: ", test_accuracy)


def NAG(layer_architecture, X_train, Y_train, X_val, Y_val, X_test, Y_test, epochs=3, activation='tanh',  weight_ini = 'He', learning_rate=0.1, beta=0.9, batch=1, weight_decay=0.0):
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
            HL, previous_store = nn.forward_propagation(X)
            nn.parameters = original_parameters  # Restore original parameters

            mini_batch_loss = nn.cross_entropy(Y, HL)

            l2_reg_loss = 0
            for l in range(1, len(nn.layers)):
                l2_reg_loss += np.sum(np.square(nn.parameters['W' + str(l)]))
            l2_reg_loss = (weight_decay / (2 * m)) * l2_reg_loss
            
            # Total loss includes both mini-batch loss and regularization loss
            epoch_loss += mini_batch_loss + l2_reg_loss

            grads = nn.backpropagation(X, Y, previous_store)
            prev_v_wb = nn.update_parameters_with_momentum_or_NAG(grads, learning_rate, beta, prev_v_wb, weight_decay, batch)

        epoch_loss /= m 

        # Training accuracy and loss
        y_pred_train, _ = nn.forward_propagation(X_train)
        train_accuracy = compute_accuracy(y_pred_train, Y_train)
        train_loss = compute_loss(Y_train, y_pred_train, nn) / m
        
        # Validation accuracy and loss
        y_pred_val, _ = nn.forward_propagation(X_val)
        val_accuracy = compute_accuracy(y_pred_val, Y_val)
        val_loss = compute_loss(Y_val, y_pred_val, nn) / X_val.shape[1]   # Normalize 
        
        # Testing accuracy and loss
        y_pred_test, _ = nn.forward_propagation(X_test)
        test_accuracy = compute_accuracy(y_pred_test, Y_test)
        test_loss = compute_loss(Y_test, y_pred_test, nn) / X_test.shape[1] # Normalize
        
        print("\nEpoch ",epoch, 'Epoch Loss : ', epoch_loss)
        print("Training loss: ", train_loss, " Training accuracy: ", train_accuracy)
        print("Validation loss: ", val_loss, " Validation accuracy: ", val_accuracy)
        print("Testing loss: ", test_loss, " Testing accuracy: ", test_accuracy)

def rmsprop(layer_architecture, X_train, Y_train, X_val, Y_val, X_test, Y_test, epochs=3, activation='tanh', weight_ini = 'He', learning_rate=0.01, beta=0.9, batch=1, epsilon=1e-6, weight_decay=0.0):
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

            HL, previous_store = nn.forward_propagation(X)
            mini_batch_loss = nn.cross_entropy(Y, HL)

            l2_reg_loss = 0
            for l in range(1, len(layer_architecture)):
                l2_reg_loss += np.sum(np.square(nn.parameters['W' + str(l)]))
            l2_reg_loss = (weight_decay / (2 * m)) * l2_reg_loss
            epoch_loss += mini_batch_loss + l2_reg_loss

            grads = nn.backpropagation(X, Y, previous_store)
            v_w_and_b = nn.update_parameters_for_RMSprop(grads, learning_rate, beta, v_w_and_b, epsilon, weight_decay, batch)


        epoch_loss /= m

        # Training accuracy and loss
        y_pred_train, _ = nn.forward_propagation(X_train)
        train_accuracy = compute_accuracy(y_pred_train, Y_train)
        train_loss = compute_loss(Y_train, y_pred_train, nn) / m
        
        # Validation accuracy and loss
        y_pred_val, _ = nn.forward_propagation(X_val)
        val_accuracy = compute_accuracy(y_pred_val, Y_val)
        val_loss = compute_loss(Y_val, y_pred_val, nn) / X_val.shape[1]  # Normalize 
        
        # Testing accuracy and loss
        y_pred_test, _ = nn.forward_propagation(X_test)
        test_accuracy = compute_accuracy(y_pred_test, Y_test)
        test_loss = compute_loss(Y_test, y_pred_test, nn) / X_test.shape[1]# Normalize
        
        print("\nEpoch ",epoch, 'Epoch Loss : ', epoch_loss)
        print("Training loss: ", train_loss, " Training accuracy: ", train_accuracy)
        print("Validation loss: ", val_loss, " Validation accuracy: ", val_accuracy)
        print("Testing loss: ", test_loss, " Testing accuracy: ", test_accuracy)

def Adam(layer_architecture, X_train, Y_train, X_val, Y_val, X_test, Y_test, epochs=3, activation='tanh',  weight_ini = 'He', learning_rate=0.001, beta1=0.9, beta2=0.999,batch=1, epsilon=1e-6, weight_decay=0.0):
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

            HL, previous_store = nn.forward_propagation(X)
            mini_batch_loss = nn.cross_entropy(Y, HL)

            l2_reg_loss = 0
            for l in range(1, len(layer_architecture)):
                l2_reg_loss += np.sum(np.square(nn.parameters['W' + str(l)]))
            l2_reg_loss = (weight_decay / (2 * m)) * l2_reg_loss
            epoch_loss += mini_batch_loss + l2_reg_loss

            grads = nn.backpropagation(X, Y, previous_store)

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

                nn.update_parameters_for_Adam(learning_rate, m_w_and_b_hat_delta_W, v_w_and_b_hat_delta_W, m_w_and_b_hat_delta_b, v_w_and_b_hat_delta_b, l, epsilon, weight_decay, batch)


        epoch_loss /= m

        # Training accuracy and loss
        y_pred_train, _ = nn.forward_propagation(X_train)
        train_accuracy = compute_accuracy(y_pred_train, Y_train)
        train_loss = compute_loss(Y_train, y_pred_train, nn) / m
        
        # Validation accuracy and loss
        y_pred_val, _ = nn.forward_propagation(X_val)
        val_accuracy = compute_accuracy(y_pred_val, Y_val)
        val_loss = compute_loss(Y_val, y_pred_val, nn) / X_val.shape[1]   # Normalize 
        
        # Testing accuracy and loss
        y_pred_test, _ = nn.forward_propagation(X_test)
        test_accuracy = compute_accuracy(y_pred_test, Y_test)
        test_loss = compute_loss(Y_test, y_pred_test, nn) / X_test.shape[1] # Normalize
        
        print("\nEpoch ",epoch, 'Epoch Loss : ', epoch_loss)
        print("Training loss: ", train_loss, " Training accuracy: ", train_accuracy)
        print("Validation loss: ", val_loss, " Validation accuracy: ", val_accuracy)
        print("Testing loss: ", test_loss, " Testing accuracy: ", test_accuracy)


def Nadam(layer_architecture, X_train, Y_train, X_val, Y_val, X_test, Y_test, epochs=3, activation='tanh',  weight_ini = 'He', learning_rate=0.01, beta1=0.9, beta2=0.999, batch=1, epsilon=1e-6, weight_decay=0.0):
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

            HL, previous_store = nn.forward_propagation(X)
            mini_batch_loss = nn.cross_entropy(Y, HL)

            l2_reg_loss = 0
            for l in range(1, len(layer_architecture)):
                l2_reg_loss += np.sum(np.square(nn.parameters['W' + str(l)]))
            l2_reg_loss = (weight_decay / (2 * m)) * l2_reg_loss
            epoch_loss += mini_batch_loss + l2_reg_loss

            grads = nn.backpropagation(X, Y, previous_store)
            m_w_and_b, v_w_and_b = nn.update_parameters_for_Nadam(m_w_and_b, v_w_and_b, beta1, beta2, learning_rate, epoch, grads, epsilon, weight_decay, m)

        epoch_loss /= m

        # Training accuracy and loss
        y_pred_train, _ = nn.forward_propagation(X_train)
        train_accuracy = compute_accuracy(y_pred_train, Y_train)
        train_loss = compute_loss(Y_train, y_pred_train, nn) / m
        
        # Validation accuracy and loss
        y_pred_val, _ = nn.forward_propagation(X_val)
        val_accuracy = compute_accuracy(y_pred_val, Y_val)
        val_loss = compute_loss(Y_val, y_pred_val, nn) / X_val.shape[1]   # Normalize 
        
        # Testing accuracy and loss
        y_pred_test, _ = nn.forward_propagation(X_test)
        test_accuracy = compute_accuracy(y_pred_test, Y_test)
        test_loss = compute_loss(Y_test, y_pred_test, nn) / X_test.shape[1] # Normalize
        
        print("\nEpoch ",epoch, 'Epoch Loss : ', epoch_loss)
        print("Training loss: ", train_loss, " Training accuracy: ", train_accuracy)
        print("Validation loss: ", val_loss, " Validation accuracy: ", val_accuracy)
        print("Testing loss: ", test_loss, " Testing accuracy: ", test_accuracy)