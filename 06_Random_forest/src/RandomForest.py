import numpy as np
from DecisionTree import DecisionTree
from collections import Counter


def bootstrap_sample(X, y):
    n_samples = X.shape[0]
    idxs = np.random.choice(n_samples, n_samples, replace=True)
    return X[idxs], y[idxs]


class RandomForest:
    def __init__(self, number_trees=100, min_samples_split=2, max_depth=100, n_features=None):
        self.n_trees = number_trees
        self.min_samples = min_samples_split
        self.max_depth = max_depth
        self.n_feats = n_features
        self.trees = []

    def fit(self, X, y):
        self.trees = []
        for _ in range(self.n_trees):
            tree = DecisionTree(
                min_samples_split=self.min_samples,
                max_depth=self.max_depth,
                n_features=self.n_feats)
            X_sample, y_sample = bootstrap_sample(X, y)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def predict(self, X):
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        tree_preds = np.swapaxes(tree_preds, 0, 1)
        y_pred = [self._most_common_label(tree_prediction) for tree_prediction in tree_preds]
        return np.array(y_pred)

    def _most_common_label(self, y):
        counter = Counter(y)
        most_common = counter.most_common(1)
        return most_common[0][0]
