import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import fashion_mnist
import keras

fashion_mnist=keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
class_labels = ['T-shirt/top', 'Trouser/pants', 'Pullover shirt', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# print(f"Number of classes in training : {np.unique(train_labels)}") 
# print(f"Number of training samples, height and width of an image : {train_images.shape}\n") 

print(f"Number of classes in testing : {np.unique(test_labels)}", ) 
print(f"Number of testing samples, height and width of an image : {test_images.shape}\n") 

print("Class label and their names given below : ")
for i in np.unique(train_labels):
    print(i," : ", class_labels[i])


def plot_one_sample_each_class(train_images, train_labels, class_labels):

    plt.figure(figsize=(10,5))
    for i in np.unique(train_labels):
        plot_label_for_each_class = np.where(train_labels == i)[0][0]
        plt.subplot(2,5,i+1)
        plt.grid(False)
        plt.imshow(train_images[plot_label_for_each_class], cmap=plt.cm.binary)
        plt.xlabel(class_labels[train_labels[plot_label_for_each_class]])
    plt.show()
    
plot_one_sample_each_class(train_images, train_labels, class_labels)