import matplotlib.pyplot as plt
from keras.datasets import fashion_mnist
import keras
import numpy as np

import wandb
wandb.login()
wandb.init(project="cs23d014_assignment_1", entity="cs23d014", name='plot_1_class_each_image')

fashion_mnist=keras.datasets.fashion_mnist
(trainX, trainy), (testX, testy) = fashion_mnist.load_data()
labels = ['T-shirt/top', 'Trouser/pants', 'Pullover shirt', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

def plot_each_class_image(trainX, trainy, labels):
    img, label = [], []
    for i in np.unique(trainy):
        plot_label_for_each_class = np.where(trainy == i)[0][0]
        img.append(trainX[plot_label_for_each_class])
        label.append(labels[trainy[plot_label_for_each_class]])
    return (img, label)
    
img, label = plot_each_class_image(trainX, trainy, labels)
run_init = wandb.init()
run_init.log({"Log images": [wandb.Image(i, caption=j) for i, j in zip(img, label)]})