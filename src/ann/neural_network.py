"""
Main Neural Network Model class
Handles forward and backward propagation loops
"""
import numpy as np

from ann.activations import ReLU, Sigmoid, Tanh
from ann.neural_layer import Linear
from ann.objective_functions import MSE, CrossEntropy
from ann.optimizers import NAG, SGD, Adam, Momentum, Nadam, RMSProp
from sklearn.metrics import f1_score


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
        self.activations = []
        input_size = getattr(cli_args, "input_size", 784)
        output_size = getattr(cli_args, "output_size", 10)
        sizes = [input_size] + cli_args.hidden_sizes + [output_size]
        
        for i in range(len(sizes) - 1):
            self.layers.append(Linear(sizes[i], sizes[i+1], weight_init=cli_args.weight_init))
            if i < len(sizes) - 2:  # Add activation for hidden layers
                if cli_args.activation == 'relu':
                    self.activations.append(ReLU())
                elif cli_args.activation == 'sigmoid':
                    self.activations.append(Sigmoid())
                elif cli_args.activation == 'tanh':
                    self.activations.append(Tanh())

                else:
                    raise ValueError(f"Unsupported activation function: {cli_args.activation}")
        #if cli_args.loss == "cross_entropy":
            #self.layers.append(Softmax())

        if cli_args.loss == 'cross_entropy':
            self.loss = CrossEntropy()
        elif cli_args.loss == 'mse':
            self.loss = MSE()
        else:
            raise ValueError(f"Unsupported loss function: {cli_args.loss}")
        
        # few parameters of optimizers which are not mentioned in the CLI arguments are hardcoded as default values

        if cli_args.optimizer == 'sgd':
            self.optimizer = SGD(cli_args.learning_rate, cli_args.weight_decay)
        elif cli_args.optimizer == 'momentum':
            self.optimizer = Momentum(cli_args.learning_rate, gamma = 0.9, weight_decay = cli_args.weight_decay)
        elif cli_args.optimizer == 'nag':         
            self.optimizer = NAG(cli_args.learning_rate, gamma = 0.9, weight_decay = cli_args.weight_decay)
        elif cli_args.optimizer == 'rmsprop':
            self.optimizer = RMSProp(cli_args.learning_rate, gamma = 0.9, eps = 1e-8, weight_decay = cli_args.weight_decay)
        elif cli_args.optimizer == 'adam':            
            self.optimizer = Adam(cli_args.learning_rate, gamma1 = 0.9, gamma2 = 0.999, eps = 1e-8, weight_decay = cli_args.weight_decay)
        elif cli_args.optimizer == 'nadam':
            self.optimizer = Nadam(cli_args.learning_rate, gamma1 = 0.9, gamma2 = 0.999, eps = 1e-8, weight_decay = cli_args.weight_decay)
        else:
            raise ValueError(f"Unsupported optimizer: {cli_args.optimizer}")
    
    def forward(self, X):
        """
        Forward propagation through all layers.
        
        Args:
            X: Input data
            
        Returns:
            Output logits
        """
        self.cached_activations = []
        for i, layer in enumerate(self.layers):
            X = layer.forward(X)
            if i < len(self.activations):
                X = self.activations[i].forward(X)
                self.cached_activations.append(X) # cache activations for dead neuron ratio calculation
        return X
    
    def backward(self, y_true, y_pred):
        """
        Backward propagation to compute gradients.
        
        Args:
            y_true: True labels
            y_pred: Predicted outputs
            
        Returns:
            return grad_w, grad_b in layers (index 0 = first layer)
        """
        self.loss.forward(y_true, y_pred)
        dZ = self.loss.backward()

        grad_W_list = []
        grad_b_list = []

        for i in reversed(range(len(self.layers))):
            if i < len(self.activations):
                dZ = self.activations[i].backward(dZ)

            layer = self.layers[i]
            dZ = layer.backward(dZ)

            # Store gradients for Linear layers
            grad_W_list.append(layer.grad_W)
            grad_b_list.append(layer.grad_b)

        # Reverse so grad_W[0] corresponds to self.layers[0]
        grad_W_list.reverse()
        grad_b_list.reverse()

        # Convert to object arrays
        self.grad_W = np.empty(len(grad_W_list), dtype=object)
        self.grad_b = np.empty(len(grad_b_list), dtype=object)

        for i, (gw, gb) in enumerate(zip(grad_W_list, grad_b_list)):
            self.grad_W[i] = gw
            self.grad_b[i] = gb

        return self.grad_W, self.grad_b
    
    def update_weights(self):
        """
        Update weights using the optimizer.
        """
        # Increment timestep once per batch (for Adam/Nadam)
        if hasattr(self.optimizer, 'step'):
            self.optimizer.step()
        for layer in self.layers:
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

                logits = self.forward(X_batch)
                loss = self.loss.forward(y_batch, logits)   # loss computed for each batch and printed at the end of each epoch
                #loss = np.mean(loss)
                self.backward(y_batch, logits) # gradients computed for each batch and weights updated after each batch
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
        f1 = f1_score(y_true_labels, y_pred_labels, average="weighted")
        return accuracy, f1
    
    def get_weights(self):
        d = {}
        for i, layer in enumerate(self.layers):
            d[f"W{i}"] = layer.W.copy()
            d[f"b{i}"] = layer.b.copy()
        return d

    def set_weights(self, weight_dict):
        for i, layer in enumerate(self.layers):
            w_key = f"W{i}"
            b_key = f"b{i}"
            if w_key in weight_dict:
                layer.W = weight_dict[w_key].copy()
            if b_key in weight_dict:
                layer.b = weight_dict[b_key].copy()