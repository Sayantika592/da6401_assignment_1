"""
Optimization Algorithms
Implements: SGD, Momentum, Adam, Nadam, etc.
"""

import numpy as np

class SGD:
    def __init__(self, learning_rate=0.01, weight_decay=0.0):
        self.lr = learning_rate
        self.wd = weight_decay
    
    def update(self, layer):
        layer.W -= self.lr * (layer.grad_W + self.wd * layer.W)
        layer.b -= self.lr * layer.grad_b

class Momentum:
    def __init__(self, learning_rate=0.01, gamma=0.9, weight_decay=0.0):
        self.lr = learning_rate
        self.gamma = gamma
        self.wd = weight_decay
        self.vW = {}
        self.vb = {}

    def update(self, layer):
        key = id(layer)

        if key not in self.vW:
            self.vW[key] = np.zeros_like(layer.W)
            self.vb[key] = np.zeros_like(layer.b)

        gW = layer.grad_W + self.wd * layer.W
        gb = layer.grad_b

        # v = γv + ηg
        self.vW[key] = self.gamma * self.vW[key] + self.lr * gW
        self.vb[key] = self.gamma * self.vb[key] + self.lr * gb

        # θ = θ - v
        layer.W -= self.vW[key]
        layer.b -= self.vb[key]  

class NAG:
    def __init__(self, learning_rate, gamma=0.9, weight_decay=0.0):
        self.lr = learning_rate
        self.gamma = gamma
        self.wd = weight_decay
        self.vW = {}
        self.vb = {}

    def update(self, layer):
        key = id(layer)

        # Initialize velocity for this layer if first time
        if key not in self.vW:
            self.vW[key] = np.zeros_like(layer.W)
            self.vb[key] = np.zeros_like(layer.b)

        # Apply weight decay
        gW = layer.grad_W + self.wd * layer.W
        gb = layer.grad_b

        # Update velocity: v = γv + ηg
        self.vW[key] = self.gamma * self.vW[key] + self.lr * gW
        self.vb[key] = self.gamma * self.vb[key] + self.lr * gb

        # NAG-style update: W = W - (γv + ηg)  --> using the current velocity for calculating the look ahead gradient
        layer.W -= (self.gamma * self.vW[key] + self.lr * gW)
        layer.b -= (self.gamma * self.vb[key] + self.lr * gb)


class RMSProp:
    def __init__(self, learning_rate, gamma=0.9, eps=1e-8, weight_decay = 0.0):
        self.lr = learning_rate
        self.gamma = gamma
        self.eps = eps
        self.wd = weight_decay
        self.vW = {}
        self.vb = {}

    def update(self, layer):
        key = id(layer)

        if key not in self.vW:
            self.vW[key] = np.zeros_like(layer.W)
            self.vb[key] = np.zeros_like(layer.b)

        gW = layer.grad_W + self.wd * layer.W
        gb = layer.grad_b

        self.vW[key] = self.gamma * self.vW[key] + (1 - self.gamma) * (gW ** 2)
        self.vb[key] = self.gamma * self.vb[key] + (1 - self.gamma) * (gb ** 2)

        layer.W -= self.lr * gW / (np.sqrt(self.vW[key]) + self.eps)
        layer.b -= self.lr * gb / (np.sqrt(self.vb[key]) + self.eps)

class Adam:
    def __init__(self, learning_rate, gamma1=0.9, gamma2=0.999, eps=1e-8, weight_decay = 0.0):
        self.lr = learning_rate
        self.gamma1 = gamma1
        self.gamma2 = gamma2
        self.wd = weight_decay
        self.eps = eps
        self.mW = {}
        self.vW = {}
        self.mb = {}
        self.vb = {}
        self.t = 0

    def update(self, layer):
        key = id(layer)

        if key not in self.mW:
            self.mW[key] = np.zeros_like(layer.W)
            self.vW[key] = np.zeros_like(layer.W)
            self.mb[key] = np.zeros_like(layer.b)
            self.vb[key] = np.zeros_like(layer.b)

        gW = layer.grad_W + self.wd * layer.W
        gb = layer.grad_b

        self.t += 1

        # Update moments
        self.mW[key] = self.gamma1 * self.mW[key] + (1 - self.gamma1) * gW
        self.mb[key] = self.gamma1 * self.mb[key] + (1 - self.gamma1) * gb

        self.vW[key] = self.gamma2 * self.vW[key] + (1 - self.gamma2) * (gW ** 2)
        self.vb[key] = self.gamma2 * self.vb[key] + (1 - self.gamma2) * (gb ** 2)

        # Bias correction
        mW_hat = self.mW[key] / (1 - self.gamma1 ** self.t)
        mb_hat = self.mb[key] / (1 - self.gamma1 ** self.t)

        vW_hat = self.vW[key] / (1 - self.gamma2 ** self.t)
        vb_hat = self.vb[key] / (1 - self.gamma2 ** self.t)

        # Update
        layer.W -= self.lr * mW_hat / (np.sqrt(vW_hat) + self.eps)
        layer.b -= self.lr * mb_hat / (np.sqrt(vb_hat) + self.eps)
        
class Nadam:
    def __init__(self, learning_rate, gamma1=0.9, gamma2=0.999, eps=1e-8, weight_decay=0.0):
        self.lr = learning_rate
        self.gamma1 = gamma1
        self.gamma2 = gamma2
        self.eps = eps
        self.wd = weight_decay
        self.mW = {}
        self.vW = {}
        self.mb = {}
        self.vb = {}
        self.t = 0

    def update(self, layer):
        key = id(layer)

        if key not in self.mW:
            self.mW[key] = np.zeros_like(layer.W)
            self.vW[key] = np.zeros_like(layer.W)
            self.mb[key] = np.zeros_like(layer.b)
            self.vb[key] = np.zeros_like(layer.b)

        self.t += 1

        gW = layer.grad_W + self.wd * layer.W
        gb = layer.grad_b

        # Moment updates
        self.mW[key] = self.gamma1 * self.mW[key] + (1 - self.gamma1) * gW
        self.mb[key] = self.gamma1 * self.mb[key] + (1 - self.gamma1) * gb

        self.vW[key] = self.gamma2 * self.vW[key] + (1 - self.gamma2) * (gW ** 2)
        self.vb[key] = self.gamma2 * self.vb[key] + (1 - self.gamma2) * (gb ** 2)

        # Bias correction
        mW_hat = self.mW[key] / (1 - self.gamma1 ** self.t)
        mb_hat = self.mb[key] / (1 - self.gamma1 ** self.t)

        vW_hat = self.vW[key] / (1 - self.gamma2 ** self.t)
        vb_hat = self.vb[key] / (1 - self.gamma2 ** self.t)

        # Nesterov term
        mW_nesterov = self.gamma1 * mW_hat + (1 - self.gamma1) * gW / (1 - self.gamma1 ** self.t)
        mb_nesterov = self.gamma1 * mb_hat + (1 - self.gamma1) * gb / (1 - self.gamma1 ** self.t)

        # Update
        layer.W -= self.lr * mW_nesterov / (np.sqrt(vW_hat) + self.eps)
        layer.b -= self.lr * mb_nesterov / (np.sqrt(vb_hat) + self.eps)