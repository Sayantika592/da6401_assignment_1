"""
Loss/Objective Functions and Their Derivatives
Implements: Cross-Entropy, Mean Squared Error (MSE)
"""
import numpy as np

class MSE:
    def forward(self, y_true, y_pred):
        """
        Mean Squared Error Loss
        L = (1/n) * Σ(y_true - y_pred)^2
        """
        self.y_true = y_true
        self.y_pred = y_pred
        self.N = y_true.shape[0]
        loss = np.sum((y_true - y_pred) ** 2)/self.N
        return loss
    
    def backward(self):
        """
        Gradient of MSE Loss w.r.t. predictions
        dL/dy_pred = (2/n) * (y_pred - y_true)
        """
        dL_dy_pred = (2 / self.N) * (self.y_pred - self.y_true)
        return dL_dy_pred
    
class CrossEntropy:
    def forward(self, y_true, y_pred):
        """
        Cross-Entropy Loss for multi-class classification
        L = -Σ(y_true * log(y_pred))
        """
        self.y_true = y_true
        self.y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        self.N = y_true.shape[0]
        loss = -np.sum(y_true * np.log(self.y_pred)) / self.N
        return loss
        
    def backward(self):
        """
        Gradient of Cross-Entropy Loss w.r.t. predictions
        dL/dy_pred = - (y_true / y_pred) / n
        """
        dL_dy_pred = (self.y_pred - self.y_true)/self.N
        return dL_dy_pred