# A probabilistic clasifier based on applying Bayes' ZTheorem with strong (naive)
# independence assumptions between features

# Review Bayes' Theorem: P(A|B) = (P(B|A)*P(A))/P(B)
    # Assuming features are mutually independent: P(y|X) = (P(X|y) * P(y))/P(X)
    # Broken down into classes: P(y|X) = (P(x1|y) * P(x2|y) * ... * P(xn|y) * P(y))/P(X)
        # Select class with highest posterior probability
            # y = argmaxy log(P(x1|y)) + log(P(x2|y)) + ... + log(P(xn|y)) + log(P(y))
    # P(y) = prior proability -> frequency of each class
    # P(xi|y) = Class conditional probaility -> Model with Gaussian
        # P(xi|y) = 1/(√(2πσy^2)) * exp(-(xi-μy)^2/(2σy^2))

# Steps 
    # Training -> Calculate mean, var, and prior frequency for each class
    # Predictions -> Claculate posterior for each class with argmax and gaussian, choose class with highest posterior proability

import numpy as np

class NaiveBayes:

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self._classes = np.unique(y)
        n_classes = len(self._classes)

        # Calculate mean, var and prior for each class
        self._mean = np.zeros((n_classes, n_features), dtype=np.float64)
        self._var = np.zeros((n_classes, n_features), dtype=np.float64)
        self._priors = np.zeros((n_classes), dtype=np.float64)

        for idx, c, in enumerate(self._classes):
            X_c = X[y==c]
            self._mean[idx, :] = X_c.mean(axis=0)
            self._var[idx, :] = X_c.var(axis=0)
            self._priors[idx] = X_c.shape[0] / float(n_samples)


    def predict(self, X): 
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)
    
    def _predict(self, x):
        posteriors = []

        # calculate posterior probability 
        for idx, c in enumerate(self._classes):
            prior = np.log(self._priors[idx])
            posterior = np.sum(np.log(self._pdf(idx, x)))
            posterior += prior
            posteriors.append(posterior)
        
        # return classs with highest posterior
        return self._classes[np.argmax(posteriors)]
    
    def _pdf(self, class_idx, x):
        mean = self._mean[class_idx]
        var = self._var[class_idx]
        num = np.exp(-(x-mean)**2 / (2*var))
        den = np.sqrt(2*np.pi * var)
        return num/den

