# Support Vector Machine - Use lienar model & try to fine a linear decision boundary
    # (hyperplane) that best seperates the data!
    # The best hyperplane - one that yields largest seperation/margin btwn both classes.
        # Choose hyperplane so that distance from it to the nearest data point on each side is maximized!

# Finding weigths - Loss function: Hinge Loss
    # l = max(0, 1-yi(w*xi-b))
    # l = {0 if y*f(x)>=1 else (1-y*f(x))}
    # After, you need to add regularization for the linear model
        # if yi*f(x)>=1 -> Ji = lambda||w||^2
            # gradient if yi*f(x) >= 1 : dJi/dwk = 2(lambda)(wk) and dJi/db = 0
        # else: Ji = lambda||w||^2 + 1 -yi(w*xi-b)
            # gradient else: dJi/dwk = 2(lambda)(wk) - yi(xik) and dJi/db = yi
    # Apply update rule
        # if yi*f(x)>=1: w = w-alpha * dw = w-alpha * 2(lambda)(w)
            # b = b-alpha * db = b
        # else: w = w-alpha * dw = w-alpha * [2(lambda)(w) - yi*xi)]
            # b = b-alpha * db = b-alpha * yi

# Steps
    # Training(Learning weights):
        # Init. weights, make sure y is in range of {-1, 1}
        # Apply update rules for n_iters
    # Predictions: 
        # Calculate u=sign(w*x-b) 
            # if >=1, then class 1, else if <= -1 then class -1

import numpy as np

class SVM: 
    
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000):
        self.lr = learning_rate
        self.lp = lambda_param 
        self.n_iters = n_iters
        self.w = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape

        # make classes have values either -1 or 1
        y_ = np.where(y <= 0, -1, 1)

        # init weights
        self.w = np.zeros(n_features) # better to randomly init.
        self.bias = 0

        # learn weights with update rule
        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                condition = y_[idx] * (np.dot(x_i, self.w)-self.bias) >= 1
                if condition:
                    self.w -= self.lr * (2*self.lp*self.w)
                else:
                    self.w -= self.lr * (2*self.lp*self.w - np.dot(x_i, y_[idx]))
                    self.bias -= self.lr * y_[idx]


    def predict(self, X):
        approx = np.dot(X, self.w) - self.bias
        return np.sign(approx)

