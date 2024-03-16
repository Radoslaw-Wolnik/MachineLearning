import numpy as np
import matplotlib.pyplot as plt
np.random.seed(123)


class kMeansCluster:
    def euclidian_distance(self, x1, x2):
        suma = np.sum((x1-x2)**2)
        return np.sqrt(suma)

    def __init__(self, K=5, max_iterations=100, plot_steps=False):
        self.X = None
        self.k = K
        self.max_iters = max_iterations
        self.plotting = plot_steps
        # lists of clusters
        self.clusters = [[] for _ in range(self.k)]
        # mean feature vector for each cluster
        self.centroids = []

    def predict(self, X):
        self.X = X
        self.n_samples, self.features = X.shape

        # initialize centroids
        random_sample_idxs = np.random.choice(self.n_samples, self.k, replace = False)
        self.centroids = [self.X[idx] for idx in random_sample_idxs]

        # optimalization
        for _ in range(self.max_iters):
            # update clusters
            self.clusters = self._create_clusters(self.centroids)
            # plot ?
            if self.plotting:
                self.plot()
            # update centroids
            centroids_old = self.centroids
            self.centroids = self._get_centroids(self.clusters)
            # check if converged - is there a change between iterations
            if self._is_converged(centroids_old, self.centroids):
                break
        # return changed centroids and clusters
        return self._get_cluster_lables(self.clusters)

    def _create_clusters(self, centroids):
        # assign samples to the closest centroids
        clusters = [[] for _ in range(self.k)]
        for idx, sample in enumerate(self.X):
            centroid_idx = self._closest_centroid(sample, centroids)
            clusters[centroid_idx].append(idx)
        return clusters

    def _closest_centroid(self, sample, centroids):
        distances = [self.euclidian_distance(sample, point) for point in centroids]
        closest_idx = np.argmin(distances)
        return closest_idx

    def _get_centroids(self, clusters):
        centroids = np.zeros((self.k, self.features))
        for cluster_idx, cluster in enumerate(clusters):
            cluster_mean = np.mean(self.X[cluster], axis=0)
            centroids[cluster_idx] = cluster_mean
        return centroids

    def _is_converged(self, centroids_old, centroids_new):
        distances = [self.euclidian_distance(centroids_old[i], centroids_new[i]) for i in range(self.k)]
        return np.sum(distances) == 0

    def _get_cluster_lables(self, clusters):
        lables = np.empty(self.n_samples)
        for cluster_idx, cluster in enumerate(clusters):
            for sample_idx in cluster:
                lables[sample_idx] = cluster_idx
        return lables

    def plot(self):
        fig, ax = plt.subplots(figsize=(12,8))

        for i, index in enumerate(self.clusters):
            point = self.X[index].T
            ax.scatter(*point)

        for point in self.centroids:
            ax.scatter(*point, marker='x', color='black', linewidths=2)

        plt.show()
