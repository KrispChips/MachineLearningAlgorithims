# KMeans - Unsupervised learning method (unlabeled data) that clusters data into k different clusters.
    # Each sample is assigned to cluster with nearest mean, and the mean (centroids) & clusters are updated during an iterative process - optimization

# Iterative Optimization Process
    # init. cluster centers (e.g. randomly)
    # repeat until converged (essentially no more change):
        # Update cluster labels: Assign point to the nearest cluster center (centroid)
        # Update cluster centers (centroids): Set center to the mean of each cluster

# Finding nearest centroids - Euclidian Distance
    # Distance btwn 2 feature vectors: d(p,q) = sqrt(Σ(pi-qi)^2)

import numpy as np
import matplotlib.pyplot as plt


def euclidian_distance(x1, x2):
    return np.sqrt(np.sum((x1-x2)**2))


class KMeans:

    def __init__(self, k=5, max_iters=100, plot_steps=False):
        self.k = k
        self.max_iters = max_iters
        self.plot_steps = plot_steps
        
        # for each cluster, we put in an empty list
            # want to store sample indices later
        self.clusters = [[] for _ in range(self.k)]

        # store centers (mean vector) for each cluster
        self.centroids = []

    def predict(self, X):
        self.X = X
        self.n_samples, self.n_features = X.shape

        # init. centroids - random
        random_sample_indicis = np.random.choice(self.n_samples, self.k, replace=False)
        self.centroids = [self.X[idx] for idx in random_sample_indicis]

        # optimize clusters 
        for _ in range(self.max_iters):
            # assign samples to closest centroids -> create clusters
            self.clusters = self._create_clusters(self.centroids)

            if self.plot_steps:
                self.plot()

            # calculate new centroids from clusters
            centroids_old = self.centroids
            self.centroids = self._get_centroids(self.clusters)

            if self._is_converged(centroids_old, self.centroids):
                break

            if self.plot_steps:
                self.plot()

        # classify samples as index of their clusters
        return self._get_cluster_labels(self.clusters)
    

    def _create_clusters(self, centroids):
        # assign samples to closest centroids
        clusters = [[] for _ in range(self.k)]
        for idx, sample in enumerate(self.X):
            centroid_idx = self._closest_centroid(sample, centroids)
            clusters[centroid_idx].append(idx)
        return clusters
    
    
    def _closest_centroid(self, sample, centroids):
        # distance of current sample to each centroid and get closest
        distances = [euclidian_distance(sample, point) for point in centroids]
        closest_idx = np.argmin(distances)
        return closest_idx
    

    def _get_centroids(self, clusters):
        # assign mean value of clusters to centroids
        centroids = np.zeros((self.k, self.n_features)) # shape has to be inner-tuple 
        for cluster_idx, cluster in enumerate(clusters):
            cluster_mean = np.mean(self.X[cluster], axis=0)
            centroids[cluster_idx] = cluster_mean
        return centroids


    def _is_converged(self, centroids_old, centroids_new):
        # check distances between old and new centroids for ALL centroids
            # if 0, then return true!
        distances = [euclidian_distance(centroids_old[i], centroids_new[i]) for i in range(self.k)]
        return sum(distances) == 0

    def _get_cluster_labels(self, clusters):
        # each sample will get label of cluster it was assigned to
        labels = np.empty(self.n_samples)
        for cluster_idx, cluster in enumerate(clusters):
            for sample_idx in cluster:
                labels[sample_idx] = cluster_idx
        return labels


    def plot(self):
        # plot steps if init. true
        fig, ax = plt.subplots(figsize=(12,8))

        for i, index in enumerate(self.clusters):
            point = self.X[index].T
            ax.scatter(*point)

        for point in self.centroids:
            ax.scatter(*point, marker="x", color="black", linewidth=2)

        plt.show()

