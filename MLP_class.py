#######################################################
# Name: Phuc H. Lam
# NetID: plam6
# Email: plam6@u.rochester.edu
#######################################################

import numpy as np
import math

# Create a class of MLP with one hidden layer
class MLP:
    #__slots__ = ('W1', 'W2', 'b1', 'b2', 'Z1', 'A1', 'Z2', 'A2' 'num_in', 'num_hid', 'num_out')

    def __init__(self, num_in, num_hid, num_out):
        self.b1 = np.zeros((num_hid, 1))
        self.b2 = np.zeros((num_out, 1))
        self.W1 = 2 * np.random.random((num_hid, num_in)) - 1
        self.W2 = 2 * np.random.random((num_out, num_hid)) - 1
        self.Z1 = np.zeros((num_hid, 1))
        self.A1 = np.zeros((num_hid, 1))
        self.Z2 = np.zeros((num_out, 1))
        self.A2 = np.zeros((num_out, 1))

    # Forward propagation
    # Evaluate network for N observation
    # X is a (num_in * N) matrix
    # A2 is a (num_out * N) matrix        
    def forward_prop(self, X):
        self.Z1 = np.matmul(self.W1, X) + self.b1
        self.A1 = np.tanh(self.Z1)
        self.Z2 = np.matmul(self.W2, self.A1) + self.b2
        self.A2 = np.exp(self.Z2)
        self.A2 = self.A2 / (np.sum(self.A2, axis=0, keepdims=True) + 1e-7)
        return self.A2
    
    # Backpropagation 
    def back_prop(self, X, Y):
        s = np.shape(X)
        N = s[1]
        Y_pred = self.forward_prop(X)
        dZ2 = Y_pred - Y 
        db2 = np.sum(dZ2, axis=1, keepdims=True) / N
        t1 = np.matmul(np.transpose(self.W2), dZ2)
        deri_tanh = 1 - np.power(self.A1, 2)
        dZ1 = np.multiply(deri_tanh, t1)
        db1 = np.sum(dZ1, axis=1, keepdims=True) / N        
        dW1 = np.matmul(dZ1, np.transpose(X)) / N
        dW2 = np.matmul(dZ2, np.transpose(self.A1)) / N
        return dW1, dW2, db1, db2

    # Gradient descent
    def grad_des(self, X, Y, learn_rate):
        dW1, dW2, db1, db2 = self.back_prop(X, Y)
        self.W1 = self.W1 - learn_rate*dW1
        self.b1 = self.b1 - learn_rate*db1
        self.W2 = self.W2 - learn_rate*dW2
        self.b2 = self.b2 - learn_rate*db2
    
    # Cross-entropy error (calculate loss)
    def cross_entropy_err(self, X, Y):
        ans = 0
        Y_pred = self.forward_prop(X)
        Y_pred = np.transpose(Y_pred)
        for n in range(Y.shape[1]):
            for k in range(Y.shape[0]):
                if (Y[k][n] * Y_pred[k][n] > 0):
                    ans = ans - Y[k][n] * math.log(Y_pred[k][n])
        return ans