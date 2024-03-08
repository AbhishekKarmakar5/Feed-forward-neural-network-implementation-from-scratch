# CS6910(Fundamentals of Deep Learning) Assignment - 1

This is a Feed forward Neural Network Implementation CS6910 Assignment

## Introduction

Following is the structure of the code implementation of the feedforward neural network:-

1) activation.py - Relu and its derivative, sigmoid and its derivative, identity and its derivative, tanh and its derivative have been implemented. This file is then called at backpropagation time for weight updates or other tasks.

2) pre_process.py - In this file, normalizing the data and one hot encoding has been performed.

3) Feedforward_neural_Network.py - This file consists of a class that contains the layer architecture, activation functions of each layer, the parameters (weights and bias), initialize the parameters, softmax activation function, cross entropy loss, forward propagation, backpropagation and update parameters with respect all different optimizers (sgd, momentum, nag, rmsprop, adam, nadam).

4) optimizers.py - This file consists of all the 6 optimizers namely - sgd, momentum, nag, rmsprop, adam, nadam. This file calls the class Feedforward_NeuralNetwork for weights and bias updates.

5) train.py - This is the main file in which various arguments are taken as input and passed as per the requirement in the neural network architecture.

6) sweep_functionality.py - In this file count=100 has been considered for wandb.agent and tried out with various different parameters to figure out the best hyperparameters which leads to best validation accuracy.

### Flexibility of the code 

- In train.py you can change the 'layer_architecture' inside the train_arguments(args) function to set the number of hidden layers and the number of neurons in each hidden layer. 

- This is dynamic architecture. The length represents no. of i/p, hidden and o/p layers and each index values in 'layer_architecture' represents the total number of neurons.


```python
layer_architecture = [X_train.shape[0], 128, 64, 32, 32, 10]
```

The neural network architecture is defined as follows:

- **Input Layer**: 784 neurons (based on the shape of the input data, X_train)
- **Hidden Layers**:
  - First Input Layer - X_train.shape[0] is 784 neurons
  - First Hidden Layer: 128 neurons
  - Second Hidden Layer: 64 neurons
  - Third Hidden Layer: 32 neurons
  - Fourth Hidden Layer: 32 neurons
- **Last Hidden (Output Layer)**: 10 neurons (no. of classes)

## Getting Started

Install the requirements.txt 
```bash
pip install -r requirements.txt
```

To run the main file
```bash
python train.py
```
In the command line, you can add the arguments (example shown below):-
```bash
python train.py -e 30 -a tanh -l cross_entropy -b 256 -o nadam -lr 0.001 -nhl 4 -sz 128 -w_i 'He Normal' -w_d 0.0005
```

Arguments to be supported:

| Name               | Default Value  | Description                                                                                      |
|--------------------|----------------|--------------------------------------------------------------------------------------------------|
| -wp, --wandb_project | cs23d014_assignment_1 | Project name  used to track experiments in Weights & Biases dashboard                             |
| -we, --wandb_entity | cs23d014         | Wandb Entity used to track experiments in the Weights & Biases dashboard.                        |
| -d, --dataset      | fashion_mnist  | choices: ["mnist", "fashion_mnist"]                                                              |
| -e, --epochs       | 30              | Number of epochs to train neural network.                                                        |
| -b, --batch_size   | 256              | Batch size used to train neural network.                                                         |
| -l, --loss         | cross_entropy  | choices: ["mean_squared_error", "cross_entropy"]                                                 |
| -o, --optimizer    | nadam            | choices: ["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"]                                  |
| -lr, --learning_rate | 0.001           | Learning rate used to optimize model parameters                                                  |
| -m, --momentum     | 0.5            | Momentum used by momentum and nag optimizers.                                                    |
| -beta, --beta      | 0.5            | Beta used by rmsprop optimizer                                                                   |
| -beta1, --beta1    | 0.9            | Beta1 used by adam and nadam optimizers.                                                         |
| -beta2, --beta2    | 0.999            | Beta2 used by adam and nadam optimizers.                                                         |
| -eps, --epsilon    | 0.000001       | Epsilon used by optimizers.                                                                      |
| -w_d, --weight_decay | 0.0005            | Weight decay used by optimizers.                                                                 |
| -w_i, --weight_init | He Normal         | choices: ["random", "Xavier"]                                                                    |
| -nhl, --num_layers | 4              | Number of hidden layers used in feedforward neural network.                                      |
| -sz, --hidden_size | 128              | Number of hidden neurons in a feedforward layer.                                                 |
| -a, --activation   | tanh        | choices: ["identity", "sigmoid", "tanh", "ReLU"]                                                 |


## Defining a new Optimizer for Neural Network

This function implements a new optimization algorithm for training feedforward neural networks.

```python
def new_optimization_algo(layer_architecture: List[int], 
                          X_train: np.ndarray, Y_train: np.ndarray, 
                          X_val: np.ndarray, Y_val: np.ndarray, 
                          X_test: np.ndarray, Y_test: np.ndarray, 
                          epochs: int, activation: str, loss: str,
                          weight_ini: str, learning_rate: float, batch: int, 
                          weight_decay: float, project: str, dataset: str) -> None:
    """
    New optimization algorithm to train a feedforward neural network.
    
    Input:
    -> layer_architecture (List[int]): List containing the number of neurons in each layer.
    -> X_train (np.ndarray): i/p training data.
    -> Y_train (np.ndarray): o/p training data.
    -> X_val (np.ndarray): i/p validation data.
    -> Y_val (np.ndarray): o/p validation data.
    -> X_test (np.ndarray): i/p test data.
    -> Y_test (np.ndarray): o/p test data.
    -> epochs (int): Number of training epochs.
    -> activation (str): Activation function to be used in hidden layers.
    -> loss (str): Loss function to be used.
    -> weight_ini (str): Initialization method for weights.
    -> learning_rate (float): Learning rate for optimization.
    -> batch (int): Batch size for mini batch gradient descent.
    -> weight_decay (float): Weight decay coefficient for regularization.
    -> project (str): Name of the project.
    -> dataset (str): Name of the dataset.
    
    Returns: None


    Create the object for Feedforward_NeuralNetwork and pass the layer_architecture, activation, weight_ini, loss
    For each epoch, call the forward_propagation(), compute_loss(), backpropagation() and update_parameters() for weights and bias updates
    Perform the Regularization in case of need.
    Calculate the Training loss, training accuracy, validation loss, validation accuracy, testing loss and testing accuracy.
    """

