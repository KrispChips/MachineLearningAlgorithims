# Training: 
# - Subset of data -> Create decision tree -> Repeat as many times as # of trees

# Testing: 
# - Given data point -> Classification: hold majority vote / Regression: mean of predictions

import numpy as np
import sys
from collections import Counter
sys.path.append("../DecisionTree/")
from DecisionTree import DecisionTree

class RandomForest:
    def __init__(self, n_trees=10, max_depth=10, min_sample_split=2, n_features=None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_sample_split = min_sample_split
        self.n_features = n_features
        self.trees = []

    def fit(self, X, y):
        self.trees = []
        for _ in range(self.n_trees):
            tree = DecisionTree(max_depth=self.max_depth,
                         min_sample_split=self.min_sample_split,
                         n_features=self.n_features)
            X_sample, y_sample = self._boostrap_samples(X, y)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def _boostrap_samples(self, X, y):
        n_samples = X.shape[0]
        indxs = np.random.choice(n_samples, n_samples, replace=True) # Samples can be selected again
        return X[indxs], y[indxs]
    
    def _most_common_label(self, y):
        counter = Counter(y)
        most_common = counter.most_common(1)[0][0]
        return most_common
    
    def predict(self, X):
        predictions = np.array([tree.predict(X) for tree in self.trees])
        # Display Ex: [[1,0,1,1], [0,0,1,1], []] <- Each inner list contains prediction for each tree
            # So 1st sample is predicted 1, 2nd sample is 0, 3rd is 1, etc. all from tree 1
            # Instead, we want list of lists where the first list has predictions for first sample from all trees
                # Ex: [[1, 0, ..., ...], [0, 0, ..., ...]] , numpy function below
        tree_preds = np.swapaxes(predictions, 0, 1)
        predictions = np.array([self._most_common_label(pred) for pred in tree_preds])
        return predictions
