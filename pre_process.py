import numpy as np

def convert_labels_to_one_hot(labels, classes):
    return np.eye(classes)[labels].T

def preprocess_data(train_images, train_labels, test_images, test_labels):
    X_train = train_images.reshape(train_images.shape[0], -1).T / 255.
    X_test = test_images.reshape(test_images.shape[0], -1).T / 255.
    
    Y_train = convert_labels_to_one_hot(train_labels, 10)
    Y_test = convert_labels_to_one_hot(test_labels, 10)
    
    return X_train, Y_train, X_test, Y_test  