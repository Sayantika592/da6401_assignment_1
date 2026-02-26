"""
Activation Functions and Their Derivatives
Implements: ReLU, Sigmoid, Tanh, Softmax
"""

import numpy as np

class Sigmoid:
    def forward(self, X):
        self.out = 1/(1+np.exp(-X))
        return self.out
    
    def backward(self, dA):
        return dA * self.out * (1-self.out)
    
class Tanh:
    def forward(self, X):
        self.out = np.tanh(X)
        return self.out
    
    def backward(self, dA):
        return dA * self.out * (1-(self.out)**2)
    
class ReLU:
    def forward(self, X):
        self.mask = X >0
        return np.maximum(0, X)
    
    def backward(self, dA):
        return dA * self.mask

class Softmax:
    def forward(self, Z):
        Z_shift = Z - np.max(Z, axis=1, keepdims=True)
        expZ = np.exp(Z_shift)
        self.out = expZ / np.sum(expZ, axis=1, keepdims=True)
        return self.out
    
    def backward(self, dA):
        batch_size, num_classes = self.out.shape
        dZ = np.zeros_like(self.out)

        for i in range(batch_size):
            s = self.out[i].reshape(-1, 1)
            jacobian = np.diagflat(s) - np.dot(s, s.T)
            dZ[i] = np.dot(jacobian, dA[i])

        return dZ