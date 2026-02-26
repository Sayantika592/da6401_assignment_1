"""
Data Loading and Preprocessing
Handles MNIST and Fashion-MNIST datasets
"""
from more_itertools import one

from keras.datasets import mnist, fashion_mnist
import numpy as np

def load_data():
    if dataset == "mnist":
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
    elif dataset == "fashion_mnist":
        (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

    
    # Pixel normalized to [0, 1] for faster and better training convergence as inputs are small
    X_train = X_train / 255.0
    X_test = X_test / 255.0

    X_train = X_train.reshape(X_train.shape[0], -1) # flatten 28x28 images to 784-dimensional vectors
    X_test = X_test.reshape(X_test.shape[0], -1)

    def one_hot_encode(labels, num_classes=10):
        onehot= np.zeros((labels.shape[0], num_classes))
        onehot[np.arange(labels.shape[0]), labels] = 1
        return onehot

    y_train = one_hot_encode(y_train)
    y_test = one_hot_encode(y_test)
    return X_train, y_train, X_test, y_test