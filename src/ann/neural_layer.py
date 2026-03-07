"""
Neural Layer Implementation
Handles weight initialization, forward pass, and gradient computation
"""
import numpy as np

class Linear:
    def __init__(self, in_features, out_features, weight_init="random"):
        if weight_init == "random":                  # random initialization
            self.W = 0.01 * np.random.randn(in_features, out_features)

        elif weight_init == "xavier":                # Xavier initialization
            limit = np.sqrt(1 / in_features)
            self.W = np.random.uniform(-limit, limit, (in_features, out_features))

        elif weight_init == "zeros":                 # Zeros initialization (for symmetry analysis)
            self.W = np.zeros((in_features, out_features))
            
        self.b = np.zeros((1, out_features))

    def forward(self, X):
        self.X = X
        return np.dot(X, self.W) + self.b
    
    def backward(self, dZ):
        # batch_size = self.X.shape[0]
        self.grad_W = np.dot(self.X.T, dZ) #/batch_size
        self.grad_b = np.sum(dZ, axis=0, keepdims=True) #/batch_size
        dX = np.dot(dZ, self.W.T)
        return dX