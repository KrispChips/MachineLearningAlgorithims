# PCA - unsupervised learning method used to reduce dimensionality of dataset
    # by transforming a large set into a lower dim. set that still contains most of the info of large set

# Goal: find a transformation such that:
        # tranformed features are linearly independent
        # dimensionality can be reduced by taking only the dimensions with highest importance
        # newly found dimensions should minimize the projection error
        # projected points should have maximum spread -> maximum variance

# Variance(X) = 1/n Σ(Xi-x̄)^2
# Covariance Matric - Indicates level to which 2 variables vary together
# Cov(X,Y) = 1/n Σ(Xi-x̄)(Yi-Ȳ)^T
# Eigenvectors - point in the direction of the max variance
# Eigenvalues - indicate the importance of its corresponding eigenvector
    # Aṽ = λṽ

# Steps:
    # Subtract mean from X
    # Calculate Cov(X, X)
    # Calculate eigenvectors and eigenvalues of the covariance matrix
    # Sort the eigenvectors according to their eigenvalues in decreasing order
    # Choose the first k eigenvectors and that will be the new k dimensions
    # Transform the original n-dimensional data points into k dimensions
        # So the projections with dot product


import numpy as np

class PCA:

    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None
        self.mean = None

    def fit(self, X): # Unsupervised so need for labels
        # mean centering - subtract mean
        self.mean = np.mean(X, axis=0)
        X -= self.mean

        # covariance - function needs samples as column so transform
        cov = np.cov(X.T)

        # eigenvectors, eigenvalues
        eigenvectors, eigenvalues = np.linalg.eig(cov)

        # eigenvectors v = [:, i] column vector -> transpose for easier calculations
        eigenvectors = eigenvectors.T

        # sort eigenvectors in decreasing order
        idxs = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idxs]
        eigenvectors = eigenvectors[idxs]

        # save the first k eigenvectors -> n_components
        self.components = eigenvectors[:self.n_components]


    def transform(self, X):
        #project data
        X -= self.mean
        return np.dot(X, self.components.T)

