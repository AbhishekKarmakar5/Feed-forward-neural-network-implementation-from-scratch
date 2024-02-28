import numpy as np
from activation import *
from Feedforward_Neural_Network import *
import wandb

# def test_accuracy(y_pred, Y_test, acc_of='Validation'):
#     print(acc_of, " accuracy :", np.mean(np.argmax(y_pred, axis=0) == np.argmax(Y_test, axis=0)))

def compute_accuracy(y_pred, Y):
    return np.mean(np.argmax(y_pred, axis=0) == np.argmax(Y, axis=0))

def compute_loss(Y, HL, nn):
    if nn.loss == 'cross_entropy':
        return nn.cross_entropy(Y, HL)
    elif nn.loss == 'mean_squared_error':
        m = Y.shape[1]
        loss = np.sum((Y - HL)**2) / m
        return loss
    else:
        print("Choose mean_squared_error OR cross_entropy")

def SGD(layer_architecture, X_train, Y_train, X_val, Y_val, X_test, Y_test, epochs=20, activation='relu', loss = 'mean_squared_error' ,weight_ini = 'He Normal', learning_rate=0.001, batch_size=1, 
        weight_decay=0.0, project="cs23d014_assignment_1", dataset='fashion_mnist'):
    nn = Feedforward_NeuralNetwork(layer_architecture, activation, weight_ini, loss)
    # to_run = 'd_'+str(dataset)+'_ep_'+str(epochs)+'_a_'+str(activation)+'_ls_'+str(loss)+'_bs_'+str(batch_size)+'_op_SGD'+'_lr_'+str(learning_rate)+'_nhl_'+str(len(layer_architecture)-2)+'_sz_'+str(layer_architecture[1])+'_w_i_'+weight_ini+'_w_d_'+str(weight_decay)
    # wandb.init(project=project, name=to_run)
    
    m = X_train.shape[1]  
    for epoch in range(epochs):
        epoch_loss = 0
        for i in range(0, m, batch_size):
            X = X_train[:, i:i+batch_size]
            Y = Y_train[:, i:i+batch_size]
            
            HL, previous_store = nn.forward_propagation(X)
            mini_batch_loss = compute_loss(Y, HL, nn)
            epoch_loss += mini_batch_loss 
            
            grads = nn.backpropagation(X, Y, previous_store)
            nn.update_parameters(grads, learning_rate, weight_decay, batch_size)
        
        epoch_loss /= m

        # Training accuracy and loss
        y_pred_train, _ = nn.forward_propagation(X_train)
        train_accuracy = compute_accuracy(y_pred_train, Y_train)
        train_loss = compute_loss(Y_train, y_pred_train, nn) #/ m
        
        # Validation accuracy and loss
        y_pred_val, _ = nn.forward_propagation(X_val)
        val_accuracy = compute_accuracy(y_pred_val, Y_val)
        val_loss = compute_loss(Y_val, y_pred_val, nn) #/ X_val.shape[1] 
        
        # Testing accuracy and loss
        y_pred_test, _ = nn.forward_propagation(X_test)
        test_accuracy = compute_accuracy(y_pred_test, Y_test)
        test_loss = compute_loss(Y_test, y_pred_test, nn) #/ X_test.shape[1]
        print("\nEpoch ",epoch, 'Epoch Loss : ', epoch_loss)
        print("Training loss: ", train_loss, " Training accuracy: ", train_accuracy)
        print("Validation loss: ", val_loss, " Validation accuracy: ", val_accuracy)
        print("Testing loss: ", test_loss, " Testing accuracy: ", test_accuracy)
        wandb.log({'Epoch': epoch, 'Training loss': train_loss, 'Training accuracy':train_accuracy, 'Validation loss': val_loss, 'Validation accuracy':val_accuracy, 'Testing loss': test_loss, 'Testing accuracy':test_accuracy})
        
        preds_class_indx = np.argmax(y_pred_test, axis=0)  
        y_true_class_indx = np.argmax(Y_test, axis=0) 
        labels = ['T-shirt/top', 'Trouser/pants', 'Pullover shirt', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']  
        wandb.log({'confusion_matrix': wandb.plot.confusion_matrix(probs=None, y_true=y_true_class_indx,preds=preds_class_indx,class_names=labels)})

def MGD(layer_architecture, X_train, Y_train, X_val, Y_val, X_test, Y_test, epochs=20, activation='tanh', loss = 'mean_squared_error', weight_ini = 'He Normal', learning_rate=0.001, beta=0.9, batch_size=1, 
        weight_decay=0.0, project="cs23d014_assignment_1", dataset='fashion_mnist'):
    nn = Feedforward_NeuralNetwork(layer_architecture, activation, weight_ini, loss)
    # to_run = 'd_'+str(dataset)+'_ep_'+str(epochs)+'_a_'+str(activation)+'_ls_'+str(loss)+'_bs_'+str(batch_size)+'_op_MGD'+'_lr_'+str(learning_rate)+'_nhl_'+str(len(layer_architecture)-2)+'_sz_'+str(layer_architecture[1])+'_w_i_'+str(weight_ini)+'_w_d_'+str(weight_decay)
    # wandb.init(project=project, name=to_run)

    m = X_train.shape[1]

    u_w_b = {} # history vectors
    for l in range(1, len(nn.layers)):
        u_w_b['delta_W' + str(l)] = np.zeros_like(nn.parameters['W' + str(l)])
        u_w_b['delta_b' + str(l)] = np.zeros_like(nn.parameters['b' + str(l)])
    
    for epoch in range(epochs):
        epoch_loss = 0
        for i in range(0, m, batch_size):
            X = X_train[:, i:i+batch_size]
            Y = Y_train[:, i:i+batch_size]
            
            HL, previous_store = nn.forward_propagation(X)
            mini_batch_loss = compute_loss(Y, HL, nn)
            epoch_loss += mini_batch_loss 
            
            grads = nn.backpropagation(X, Y, previous_store)
            nn.update_parameters_with_momentum_or_NAG(grads, learning_rate, beta, u_w_b, weight_decay, batch_size)
        
        epoch_loss /= m

        # Training accuracy and loss
        y_pred_train, _ = nn.forward_propagation(X_train)
        train_accuracy = compute_accuracy(y_pred_train, Y_train)
        train_loss = compute_loss(Y_train, y_pred_train, nn) #/ m
        
        # Validation accuracy and loss
        y_pred_val, _ = nn.forward_propagation(X_val)
        val_accuracy = compute_accuracy(y_pred_val, Y_val)
        val_loss = compute_loss(Y_val, y_pred_val, nn) #/ X_val.shape[1] 
        
        # Testing accuracy and loss
        y_pred_test, _ = nn.forward_propagation(X_test)
        test_accuracy = compute_accuracy(y_pred_test, Y_test)
        test_loss = compute_loss(Y_test, y_pred_test, nn) #/ X_test.shape[1]
        
        print("\nEpoch ",epoch, 'Epoch Loss : ', epoch_loss)
        print("Training loss: ", train_loss, " Training accuracy: ", train_accuracy)
        print("Validation loss: ", val_loss, " Validation accuracy: ", val_accuracy)
        print("Testing loss: ", test_loss, " Testing accuracy: ", test_accuracy)
        wandb.log({'Epoch': epoch, 'Training loss': train_loss, 'Training accuracy':train_accuracy, 'Validation loss': val_loss, 'Validation accuracy':val_accuracy, 'Testing loss': test_loss, 'Testing accuracy':test_accuracy})
        
        preds_class_indx = np.argmax(y_pred_test, axis=0)  
        y_true_class_indx = np.argmax(Y_test, axis=0) 
        labels = ['T-shirt/top', 'Trouser/pants', 'Pullover shirt', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']  
        wandb.log({'confusion_matrix': wandb.plot.confusion_matrix(probs=None, y_true=y_true_class_indx,preds=preds_class_indx,class_names=labels)})


def NAG(layer_architecture, X_train, Y_train, X_val, Y_val, X_test, Y_test, epochs=20, activation='tanh', loss = 'mean_squared_error' ,  weight_ini = 'He Normal', learning_rate=0.1, beta=0.9, batch_size=1, 
        weight_decay=0.0, project="cs23d014_assignment_1", dataset='fashion_mnist'):
    nn = Feedforward_NeuralNetwork(layer_architecture, activation, weight_ini, loss)
    # to_run = 'd_'+str(dataset)+'_ep_'+str(epochs)+'_a_'+str(activation)+'_ls_'+str(loss)+'_bs_'+str(batch_size)+'_op_NAG'+'_lr_'+str(learning_rate)+'_nhl_'+str(len(layer_architecture)-2)+'_sz_'+str(layer_architecture[1])+'_w_i_'+str(weight_ini)+'_w_d_'+str(weight_decay)
    # wandb.init(project=project, name=to_run)

    m = X_train.shape[1]

    prev_v_wb = {}  # history
    for l in range(1, len(nn.layers)):
        prev_v_wb['delta_W' + str(l)] = np.zeros_like(nn.parameters['W' + str(l)])
        prev_v_wb['delta_b' + str(l)] = np.zeros_like(nn.parameters['b' + str(l)])

    for epoch in range(epochs):
        epoch_loss = 0
        for i in range(0, m, batch_size):
            X = X_train[:, i:i+batch_size]
            Y = Y_train[:, i:i+batch_size]

            # Look-ahead step
            v_w_and_v_b = {}
            for l in range(1, len(nn.layers)):
                v_w_and_v_b["W" + str(l)] = nn.parameters["W" + str(l)] - beta * prev_v_wb["delta_W" + str(l)]
                v_w_and_v_b["b" + str(l)] = nn.parameters["b" + str(l)] - beta * prev_v_wb["delta_b" + str(l)]
            
            # Use v_w_and_v_b parameters for forward propagation
            nn.parameters, original_parameters = v_w_and_v_b, nn.parameters
            HL, previous_store = nn.forward_propagation(X)
            nn.parameters = original_parameters  # Restore original parameters

            mini_batch_loss = compute_loss(Y, HL, nn)
            epoch_loss += mini_batch_loss

            grads = nn.backpropagation(X, Y, previous_store)
            prev_v_wb = nn.update_parameters_with_momentum_or_NAG(grads, learning_rate, beta, prev_v_wb, weight_decay, batch_size)

        epoch_loss /= m 

        # Training accuracy and loss
        y_pred_train, _ = nn.forward_propagation(X_train)
        train_accuracy = compute_accuracy(y_pred_train, Y_train)
        train_loss = compute_loss(Y_train, y_pred_train, nn) #/ m
        
        # Validation accuracy and loss
        y_pred_val, _ = nn.forward_propagation(X_val)
        val_accuracy = compute_accuracy(y_pred_val, Y_val)
        val_loss = compute_loss(Y_val, y_pred_val, nn) #/ X_val.shape[1]  
        
        # Testing accuracy and loss
        y_pred_test, _ = nn.forward_propagation(X_test)
        test_accuracy = compute_accuracy(y_pred_test, Y_test)
        test_loss = compute_loss(Y_test, y_pred_test, nn) #/ X_test.shape[1] 
        
        print("\nEpoch ",epoch, 'Epoch Loss : ', epoch_loss)
        print("Training loss: ", train_loss, " Training accuracy: ", train_accuracy)
        print("Validation loss: ", val_loss, " Validation accuracy: ", val_accuracy)
        print("Testing loss: ", test_loss, " Testing accuracy: ", test_accuracy)
        wandb.log({'Epoch': epoch, 'Training loss': train_loss, 'Training accuracy':train_accuracy, 'Validation loss': val_loss, 'Validation accuracy':val_accuracy, 'Testing loss': test_loss, 'Testing accuracy':test_accuracy})
        
        preds_class_indx = np.argmax(y_pred_test, axis=0)  
        y_true_class_indx = np.argmax(Y_test, axis=0) 
        labels = ['T-shirt/top', 'Trouser/pants', 'Pullover shirt', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']  
        wandb.log({'confusion_matrix': wandb.plot.confusion_matrix(probs=None, y_true=y_true_class_indx,preds=preds_class_indx,class_names=labels)})

def rmsprop(layer_architecture, X_train, Y_train, X_val, Y_val, X_test, Y_test, epochs=3, activation='tanh', loss = 'mean_squared_error' , weight_ini = 'He Normal', learning_rate=0.01, beta=0.9, batch_size=1, 
            epsilon=1e-6, weight_decay=0.0, project="cs23d014_assignment_1", dataset='fashion_mnist'):
    nn = Feedforward_NeuralNetwork(layer_architecture, activation, weight_ini, loss)
    # to_run = 'd_'+str(dataset)+'_ep_'+str(epochs)+'_a_'+str(activation)+'_ls_'+str(loss)+'_bs_'+str(batch_size)+'_op_RMSprop'+'_lr_'+str(learning_rate)+'_nhl_'+str(len(layer_architecture)-2)+'_sz_'+str(layer_architecture[1])+'_w_i_'+str(weight_ini)+'_w_d_'+str(weight_decay)
    # wandb.init(project=project, name=to_run)
    m = X_train.shape[1]

    v_w_and_b = {}  # squared gradients summation for RMSprop
    for l in range(1, len(nn.layers)):
        v_w_and_b['delta_W' + str(l)] = np.zeros_like(nn.parameters['W' + str(l)])
        v_w_and_b['delta_b' + str(l)] = np.zeros_like(nn.parameters['b' + str(l)])

    for epoch in range(epochs):
        epoch_loss = 0
        for i in range(0, m, batch_size):
            X = X_train[:, i:i+batch_size]
            Y = Y_train[:, i:i+batch_size]

            HL, previous_store = nn.forward_propagation(X)
            mini_batch_loss = compute_loss(Y, HL, nn)
            epoch_loss += mini_batch_loss 

            grads = nn.backpropagation(X, Y, previous_store)
            v_w_and_b = nn.update_parameters_for_RMSprop(grads, learning_rate, beta, v_w_and_b, epsilon, weight_decay, batch_size)

        epoch_loss /= m

        # Training accuracy and loss
        y_pred_train, _ = nn.forward_propagation(X_train)
        train_accuracy = compute_accuracy(y_pred_train, Y_train)
        train_loss = compute_loss(Y_train, y_pred_train, nn) #/ m
        
        # Validation accuracy and loss
        y_pred_val, _ = nn.forward_propagation(X_val)
        val_accuracy = compute_accuracy(y_pred_val, Y_val)
        val_loss = compute_loss(Y_val, y_pred_val, nn) #/ X_val.shape[1]  
        
        # Testing accuracy and loss
        y_pred_test, _ = nn.forward_propagation(X_test)
        test_accuracy = compute_accuracy(y_pred_test, Y_test)
        test_loss = compute_loss(Y_test, y_pred_test, nn) #/ X_test.shape[1]
        
        print("\nEpoch ",epoch, 'Epoch Loss : ', epoch_loss)
        print("Training loss: ", train_loss, " Training accuracy: ", train_accuracy)
        print("Validation loss: ", val_loss, " Validation accuracy: ", val_accuracy)
        print("Testing loss: ", test_loss, " Testing accuracy: ", test_accuracy)
        wandb.log({'Epoch': epoch, 'Training loss': train_loss, 'Training accuracy':train_accuracy, 'Validation loss': val_loss, 'Validation accuracy':val_accuracy, 'Testing loss': test_loss, 'Testing accuracy':test_accuracy})
        
        preds_class_indx = np.argmax(y_pred_test, axis=0)  
        y_true_class_indx = np.argmax(Y_test, axis=0) 
        labels = ['T-shirt/top', 'Trouser/pants', 'Pullover shirt', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']  
        wandb.log({'confusion_matrix': wandb.plot.confusion_matrix(probs=None, y_true=y_true_class_indx,preds=preds_class_indx,class_names=labels)})

def Adam(layer_architecture, X_train, Y_train, X_val, Y_val, X_test, Y_test, epochs=3, activation='tanh', loss = 'mean_squared_error' ,  weight_ini = 'He Normal', learning_rate=0.001, beta1=0.9, beta2=0.999,batch_size=1, 
         epsilon=1e-6, weight_decay=0.0, project="cs23d014_assignment_1", dataset='fashion_mnist'):
    nn = Feedforward_NeuralNetwork(layer_architecture, activation, weight_ini, loss)
    # to_run = 'd_'+str(dataset)+'_ep_'+str(epochs)+'_a_'+str(activation)+'_ls_'+str(loss)+'_bs_'+str(batch_size)+'_op_Adam'+'_lr_'+str(learning_rate)+'_nhl_'+str(len(layer_architecture)-2)+'_sz_'+str(layer_architecture[1])+'_w_i_'+str(weight_ini)+'_w_d_'+str(weight_decay)
    # wandb.init(project=project, name=to_run)

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
        for i in range(0, m, batch_size):
            X = X_train[:, i:i+batch_size]
            Y = Y_train[:, i:i+batch_size]

            HL, previous_store = nn.forward_propagation(X)
            mini_batch_loss = compute_loss(Y, HL, nn)
            epoch_loss += mini_batch_loss 

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

                nn.update_parameters_for_Adam(learning_rate, m_w_and_b_hat_delta_W, v_w_and_b_hat_delta_W, m_w_and_b_hat_delta_b, v_w_and_b_hat_delta_b, l, epsilon, weight_decay, batch_size)


        epoch_loss /= m

        # Training accuracy and loss
        y_pred_train, _ = nn.forward_propagation(X_train)
        train_accuracy = compute_accuracy(y_pred_train, Y_train)
        train_loss = compute_loss(Y_train, y_pred_train, nn) #/ m
        
        # Validation accuracy and loss
        y_pred_val, _ = nn.forward_propagation(X_val)
        val_accuracy = compute_accuracy(y_pred_val, Y_val)
        val_loss = compute_loss(Y_val, y_pred_val, nn) #/ X_val.shape[1] 
        
        # Testing accuracy and loss
        y_pred_test, _ = nn.forward_propagation(X_test)
        test_accuracy = compute_accuracy(y_pred_test, Y_test)
        test_loss = compute_loss(Y_test, y_pred_test, nn) #/ X_test.shape[1] 
        
        print("\nEpoch ",epoch)
        print("Training loss: ", train_loss, " Training accuracy: ", train_accuracy)
        print("Validation loss: ", val_loss, " Validation accuracy: ", val_accuracy)
        print("Testing loss: ", test_loss, " Testing accuracy: ", test_accuracy)
        wandb.log({'Epoch': epoch, 'Training loss': train_loss, 'Training accuracy':train_accuracy, 'Validation loss': val_loss, 'Validation accuracy':val_accuracy, 'Testing loss': test_loss, 'Testing accuracy':test_accuracy})
        
        preds_class_indx = np.argmax(y_pred_test, axis=0)  
        y_true_class_indx = np.argmax(Y_test, axis=0) 
        labels = ['T-shirt/top', 'Trouser/pants', 'Pullover shirt', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']  
        wandb.log({'confusion_matrix': wandb.plot.confusion_matrix(probs=None, y_true=y_true_class_indx,preds=preds_class_indx,class_names=labels)})

def Nadam(layer_architecture, X_train, Y_train, X_val, Y_val, X_test, Y_test, epochs=3, activation='tanh', loss = 'mean_squared_error' ,  weight_ini = 'He Normal', learning_rate=0.01, beta1=0.9, beta2=0.999, batch_size=1, 
          epsilon=1e-6, weight_decay=0.0, project="cs23d014_assignment_1", dataset='fashion_mnist'):
    nn = Feedforward_NeuralNetwork(layer_architecture, activation, weight_ini, loss)
    # to_run = 'd_'+str(dataset)+'_ep_'+str(epochs)+'_a_'+str(activation)+'_ls_'+str(loss)+'_bs_'+str(batch_size)+'_op_Nadam'+'_lr_'+str(learning_rate)+'_nhl_'+str(len(layer_architecture)-2)+'_sz_'+str(layer_architecture[1])+'_w_i_'+str(weight_ini)+'_w_d_'+str(weight_decay)
    # wandb.init(project=project, name=to_run)
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
        for i in range(0, m, batch_size):
            X = X_train[:, i:i+batch_size]
            Y = Y_train[:, i:i+batch_size]

            HL, previous_store = nn.forward_propagation(X)
            mini_batch_loss = compute_loss(Y, HL, nn)
            epoch_loss += mini_batch_loss

            grads = nn.backpropagation(X, Y, previous_store)
            m_w_and_b, v_w_and_b = nn.update_parameters_for_Nadam(m_w_and_b, v_w_and_b, beta1, beta2, learning_rate, epoch, grads, epsilon, weight_decay, m)

        epoch_loss /= m

        # Training accuracy and loss
        y_pred_train, _ = nn.forward_propagation(X_train)
        train_accuracy = compute_accuracy(y_pred_train, Y_train)
        train_loss = compute_loss(Y_train, y_pred_train, nn) # / m
        
        # Validation accuracy and loss
        y_pred_val, _ = nn.forward_propagation(X_val)
        val_accuracy = compute_accuracy(y_pred_val, Y_val)
        val_loss = compute_loss(Y_val, y_pred_val, nn) # / X_val.shape[1]
        
        # Testing accuracy and loss
        y_pred_test, _ = nn.forward_propagation(X_test)
        test_accuracy = compute_accuracy(y_pred_test, Y_test)
        test_loss = compute_loss(Y_test, y_pred_test, nn) # / X_test.shape[1]
        
        print("\nEpoch ",epoch)
        print("Training loss: ", train_loss, " Training accuracy: ", train_accuracy)
        print("Validation loss: ", val_loss, " Validation accuracy: ", val_accuracy)
        print("Testing loss: ", test_loss, " Testing accuracy: ", test_accuracy)
        wandb.log({'Epoch': epoch, 'Training loss': train_loss, 'Training accuracy':train_accuracy, 'Validation loss': val_loss, 'Validation accuracy':val_accuracy, 'Testing loss': test_loss, 'Testing accuracy':test_accuracy})
        
        preds_class_indx = np.argmax(y_pred_test, axis=0)  
        y_true_class_indx = np.argmax(Y_test, axis=0) 
        labels = ['T-shirt/top', 'Trouser/pants', 'Pullover shirt', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']  
        wandb.log({'confusion_matrix': wandb.plot.confusion_matrix(probs=None, y_true=y_true_class_indx,preds=preds_class_indx,class_names=labels)})