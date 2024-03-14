import numpy as np
from math import sqrt
from collections import Counter

def euclidian_distance(v1, v2):
    # determines a distance between two vectors
    assert len(v1) != len(v2), "Different dimensions of vectors"
    sum_vectors = 0
    for i in range(len(v1)):
        sum_vectors += (v1[i] - v2[i]) ** 2
    return sqrt(sum_vectors)

def np_euclidian(v1, v2):
    return np.sqrt(np.sum((v1-v2)**2))

class KNN:

    def __init__(self, k = 3):
        # k =  k nearest neighbours to look at
        self.y_train = None
        self.X_train = None
        self.k = k

    def fit(self, X, y):
        # fit training samples and training lables
        # there are already fitted
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        # can have multiple samples in X so we pass each sample to _predict
        predict_lables =[self._predict(x) for x in X]
        return np.array(predict_lables)

    def _predict(self, x):
        # gets one sample form predict
        # compute the distances
        distances = [np_euclidian(x, x_train) for x_train in self.X_train]
        # get k nearest samples and their labels
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_lables = [self.y_train[v] for v in k_indices]
        # majority vote - most common label
        most_common = Counter(k_nearest_lables).most_common(1)
        return most_common[0][0]
