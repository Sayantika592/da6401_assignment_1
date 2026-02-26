"""
Main Neural Network Model class
Handles forward and backward propagation loops
"""
import numpy as np

from ann.activations import ReLU, Sigmoid, Softmax, Tanh
from ann.neural_layer import Linear
from ann.objective_functions import MSE, CrossEntropy
from ann.optimizers import NAG, SGD, Adam, Momentum, Nadam, RMSProp


class NeuralNetwork:
    """
    Main model class that orchestrates the neural network training and inference.
    """
    
    def __init__(self, cli_args):
        """
        Initialize the neural network.

        Args:
            cli_args: Command-line arguments for configuring the network
        """
        self.cli_args = cli_args
        self.layers = []
        sizes = [cli_args.input_size] + cli_args.hidden_sizes + [cli_args.output_size]

        for i in range(len(sizes) - 1):
            self.layers.append(Linear(sizes[i], sizes[i+1], weight_init=cli_args.weight_init))
            if i < len(sizes) - 2:  # Add activation for hidden layers
                if cli_args.activation == 'relu':
                    self.layers.append(ReLU())
                elif cli_args.activation == 'sigmoid':
                    self.layers.append(Sigmoid())
                elif cli_args.activation == 'tanh':
                    self.layers.append(Tanh())
        self.layers.append(Softmax())

        if cli_args.loss == 'cross_entropy':
            self.loss = CrossEntropy()
        elif cli_args.loss == 'mse':
            self.loss = MSE()
        
        if cli_args.optimizer == 'sgd':
            self.optimizer = SGD(cli_args.learning_rate)
        if cli_args.optimizer == 'momentum':
            self.optimizer = Momentum(cli_args.learning_rate, cli_args.momentum)
        if cli_args.optimizer == 'nag':         
            self.optimizer = NAG(cli_args.learning_rate, cli_args.momentum)
        if cli_args.optimizer == 'rmsprop':
            self.optimizer = RMSProp(cli_args.learning_rate, cli_args.rms_decay)
        if cli_args.optimizer == 'adam':            
            self.optimizer = Adam(cli_args.learning_rate, cli_args.adam_beta1, cli_args.adam_beta2)
        if cli_args.optimizer == 'nadam':
            self.optimizer = Nadam(cli_args.learning_rate, cli_args.adam_beta1, cli_args.adam_beta2)
    
    def forward(self, X):
        """
        Forward propagation through all layers.
        
        Args:
            X: Input data
            
        Returns:
            Output logits
        """
        for layers in self.layers:
            X = layers.forward(X)
        return X
    
    def backward(self, y_true, y_pred):
        """
        Backward propagation to compute gradients.
        
        Args:
            y_true: True labels
            y_pred: Predicted outputs
            
        Returns:
            return grad_w, grad_b in layers
        """
        loss = self.loss.forward(y_true, y_pred)
        dZ = self.loss.backward()

        for layer in reversed(self.layers):
            if isinstance(layer, Softmax):
                continue
            dZ = layer.backward(dZ)

        return loss
    
    def update_weights(self):
        """
        Update weights using the optimizer.
        """
        for layer in self.layers:
            if isinstance(layer, Linear):
                self.optimizer.update(layer)
    
    def train(self, X_train, y_train, epochs, batch_size):
        """
        Train the network for specified epochs.
        """
        N = X_train.shape[0]

        for epoch in range(epochs):
            for i in range(0, N, batch_size):
                X_batch = X_train[i:i+batch_size]
                y_batch = y_train[i:i+batch_size]

                y_pred = self.forward(X_batch)
                loss = self.backward(y_batch, y_pred)
                self.update_weights()
            print(f"Epoch {epoch+1}, Loss: {loss:.4f}")
    
    def evaluate(self, X, y):
        """
        Evaluate the network on given data.
        """
        y_pred = self.forward(X)
        y_pred_labels = np.argmax(y_pred, axis=1)
        y_true_labels = np.argmax(y, axis=1)
        accuracy = np.mean(y_pred_labels == y_true_labels)
        return accuracy
