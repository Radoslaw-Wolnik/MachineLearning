import numpy as np


class LDA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.linear_discriminants = None

    def fit(self, X, y):
        n_features = X.shape[1]
        class_labels = np.unique(y)

        # calculate both scatter matrixes - w and b
        mean_overall = np.mean(X, axis=0)
        S_W = np.zeros((n_features, n_features))  # eg 4 x 4
        S_B = np.zeros((n_features, n_features))  # eg 4 x 4
        for c in class_labels:
            x_c = X[y == c]
            mean_c = np.mean(x_c, axis=0)
            # 4 x n_c   *   n_c x 4   = 4 x 4
            S_W += (x_c - mean_c).T.dot(x_c - mean_c)

            n_c = x_c.shape[0]
            mean_diff = (mean_c - mean_overall).reshape(n_features, 1)
            S_B += n_c * (mean_diff).dot(mean_diff.T)
        A = np.linalg.inv(S_W).dot(S_B)
        eigenvalues, eigenvectors = np.linalg.eig(A)
        eigenvectors = eigenvectors.T
        idxs = np.argsort(abs(eigenvalues))[::-1]
        eigenvalues = eigenvalues[idxs]
        eigenvectors = eigenvectors[idxs]
        self.linear_discriminants = eigenvectors[0 : self.n_components]

    def transform(self, X):
        # project data
        return np.dot(X, self.linear_discriminants.T)
