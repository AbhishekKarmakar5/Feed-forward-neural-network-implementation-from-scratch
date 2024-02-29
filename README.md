# CS6910(Fundamentals of Deep Learning) Assignment - 1

This is a Feed forward Neural Network Implementation CS6910 Assignment

## Introduction

Following is the structure of the code implementation of neural network:-

1) activation.py - relu and its derivative, sigmoid and its derivative, identity and its derivative, tanh and its derivative has been implemented. This file is then called on backpropagation time for weight updates or any other tasks.

2) pre_process.py - In this file, normalizing the data and one hot encoding has been performed.

3) Feedforward_neural_Network.py - This file consists of a class that contains the layer architecture, activation functions of each layer, the parameters (weights and bias), initialize the parameters, softmax activation function, cross entropy loss, forward propagation, backpropagation and update parameters with respect all different optimizers (sgd, momentum, nag, rmsprop, adam, nadam).

4) optimizers.py - This file consists of all the 6 optimizers namely - sgd, momentum, nag, rmsprop, adam, nadam. This file calls the class Feedforward_NeuralNetwork for weights and bias updates.

5) train.py - This is the main file in which various arguments are taken as input and passed as per the requirement in the neural network architecture.

6) sweep_functionality.py - In this file count=100 has been considered for wandb.agent and tried out with various different parameters to figure out the best hyperparameters which leads to best validation accuracy.

In the command line, you can add the arguments (example shown below) :-
python train.py --epochs 30 -o nadam -lr 0.0001 -nhl 3 -sz 32

## Getting Started

Install the requirements.txt 
```bash
pip install -r requirements.txt
```

To run the main file
```bash
python train.py
```
