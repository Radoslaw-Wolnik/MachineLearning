import numpy as np
from collections import Counter


def entropy(y):
    hist = np.bincount(y)
    ps = hist / len(y)
    return - np.sum([p * np.log2(p) for p in ps if p > 0])


class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf(self):
        return self.value is not None


class DecisionTree:
    def __init__(self, min_samples_split=2, max_depth=100, n_features=None):
        self.min_samples = min_samples_split
        self.max_depth = max_depth
        self.n_feats = n_features
        self.root = None

    def fit(self, X, y):
        # grow tree
        self.n_feats = X.shape[1] if not self.n_feats else min(self.n_feats, X.shape[1])
        self.root = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        n_lables = len(np.unique(y))
        # stopping criteria:
        if (depth >= self.max_depth
                or n_lables == 1
                or n_samples < self.min_samples):
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        feat_idxs = np.random.choice(n_features, self.n_feats, replace=False)
        # greedy search
        best_feat, best_threshold = self._best_criteria(X, y, feat_idxs)
        left_idxs, right_idxs = self._split(X[:, best_feat], best_threshold)
        left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth + 1)
        right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth + 1)
        return Node(best_feat, best_threshold, left, right)

    def _best_criteria(self, X, y, feat_idxs):
        best_gain = -1
        split_idx, split_threshold = None, None
        for feat_idx in feat_idxs:
            X_column = X[:, feat_idx]
            thresholds = np.unique(X_column)
            for threshold in thresholds:
                gain = self._information_gain(y, X_column, threshold)

                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_threshold = threshold

        return split_idx, split_threshold

    def _information_gain(self, y, X_column, split_threshold):
        # calculate parent entropy
        parent_entropy = entropy(y)
        # generate split
        left_idxs, right_idxs = self._split(X_column, split_threshold)
        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0
        # weighted avg child entropy
        n = len(y)
        n_l, n_r = len(left_idxs), len(right_idxs)
        entropy_left = entropy(y[left_idxs])
        entropy_right = entropy(y[right_idxs])
        entropy_children = (n_l / n) * entropy_left + (n_r / n) * entropy_right
        # return information gain
        return parent_entropy - entropy_children

    def _split(self, X_column, split_threshold):
        left_idxs = np.argwhere(X_column <= split_threshold).flatten()
        right_idxs = np.argwhere(X_column > split_threshold).flatten()
        return left_idxs, right_idxs

    def _most_common_label(self, y):
        counter = Counter(y)
        most_common = counter.most_common(1)
        return most_common[0][0]

    def predict(self, X):
        # traversing tree to determine the label
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x, node):
        # check if we reached leaf node
        if node.is_leaf():
            return node.value
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)
