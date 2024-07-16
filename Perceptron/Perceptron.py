# Perceptron - Algorithm for supervised learning of binary classifiers
    # Seen as a single unit of an artifical neural network and is also the Prototype for Neural Nets

# Single Unit -> Single Layer Perceptron: Can learn only linearly seperable patterns
# Multilayer Perceptron: Can learn more compelx patterns
    # Inspired by Neurons -> Simplified model of biological neurons and simulates behavior of one cell
        # Single-layer neural network with the unti step function as an activation function
            # Ex: Inputs -> Weights -> Net input function -> Activation Function(1 if fires, else 0) -> Output

# Linear Model -> f(x) = w^Tx + b
# Activation function -> Unit step function
    # g(z) = {1 if z>=0 , 0 otherwise}
# Approximation (class label): yhat = g(f(x)) = g(w^Tx + b)
# Need to learn weights -> Perceptron Update Rule:
    # For each training sample xi:  
        # w = w + delta w
        # b = b + delta b
        # delta w = alpha * (yi-yhati) * xi
        # delta b = alpha * (yi-yhati)
            # alpha is learning rate (btwn 0 and 1)
    # Point of Rule: Weights are pushed towards positive/negative target class in case of missclassication

# Steps: 
    # Init. weights
    # For each sample calculate yhat=g(f(x))
        # Apply update rule

    # Predictions
        # Calculate yhat = g(f(x))

import numpy as np

def unit_step_func(x):
    return np.where(x > 0, 1, 0)

class Perceptron:
    def __init__(self, alpha=0.01, n_iters=1000):
        self.lr = alpha
        self.n_iters = n_iters
        self.activation = unit_step_func
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape

        #init weights
        self.weights = np.zeros(n_features) # Better way -> Randomly init.
        self.bias = 0

        y_ = np.where(y > 0, 1, 0)
        
        #learn weights
        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                linear_output = np.dot(x_i, self.weights) + self.bias
                y_hat = self.activation(linear_output)

                # Perceptron Update Rule
                update = self.lr * (y_[idx] - y_hat)
                self.weights += update*x_i
                self.bias += update


    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        y_hat = self.activation(linear_output)
        return y_hat
