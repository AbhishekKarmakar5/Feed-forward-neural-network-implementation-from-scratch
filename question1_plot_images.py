import matplotlib.pyplot as plt
from keras.datasets import fashion_mnist

fashion_mnist=keras.datasets.fashion_mnist
(trainX, trainy), (testX, testy) = fashion_mnist.load_data()
labels = ['T-shirt/top', 'Trouser/pants', 'Pullover shirt', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

def plot_each_class_image(trainX, trainy, labels):

    plt.figure(figsize=(10,5))
    for i in np.unique(trainy):
        plot_label_for_each_class = np.where(trainy == i)[0][0]
        plt.subplot(2,5,i+1)
        plt.grid(False)
        plt.imshow(trainX[plot_label_for_each_class], cmap=plt.cm.binary)
        plt.xlabel(labels[trainy[plot_label_for_each_class]])
    plt.show()
    
plot_each_class_image(trainX, trainy, labels)