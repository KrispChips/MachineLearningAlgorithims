import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from KMeans import KMeans

np.random.seed(42) # sake of making it reproducable for cross refernce

X,y = make_blobs(centers=3, n_samples=500, n_features=2, shuffle=True, random_state=123)
print(f"X Shape: {X.shape}")

clusters = len(np.unique(y))
print(f"Clusters: {clusters}")

clf = KMeans(k=clusters, max_iters=150, plot_steps=True)
y_pred = clf.predict(X)

clf.plot()
