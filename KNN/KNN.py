import numpy as np
from collections import Counter

def euclidian_distance(x1, x2):
    distance = np.sqrt(np.sum((x1-x2)**2))
    return distance

class KNN:
    def __init__(self, k):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        predictions = [self._predict(x) for x in X]
        return predictions
    
    def _predict(self, x):
        # Compute distances
        distances = [euclidian_distance(x, x_train) for x_train in self.X_train]
        # Get closest K
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
            # np.argsort -> Gives you the original indeces of the array after sorting
                # So the first 3 after np.argsort would be the indeces of the 3 closest values from the given array

        # Label w/ majority vote
        majority = Counter(k_nearest_labels).most_common()
        return majority[0][0]
