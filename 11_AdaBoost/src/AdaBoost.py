import numpy as np


# def alpha(self):
    # how good is our prediction
#     pass

class DecisionStump:
    def __init__(self):
        self.polarity = 1
        self.feature_idx = None
        self.threshold = None
        self.alpha = None

    def predict(self, X):
        n_samples = X.shape[0]
        x_column = X[:, self.feature_idx]
        predictions = np.ones(n_samples)
        if self.polarity == 1:
            # predictions = [x_column < self.threshold] = -1
            predictions = np.where(x_column < self.threshold, -1, 1)
        else:
            predictions = np.where(x_column > self.threshold, -1, 1)
        return predictions

class AdaBoost:
    def __init__(self, n_clf=5):
        self.n_clf = n_clf
        self.clfs = []

    def fit(self, X, y):
        n_samples, features = X.shape
        # initialize weights
        w = np.full(n_samples, (1/n_samples))

        self.clfs = []
        for _ in range(self.n_clf):
            # greedy search - iterate through all features and all thresholds
            clf = DecisionStump()
            # very high number
            min_error = float('inf')
            error = 0
            for feature_i in range(features):
                x_column = X[:, feature_i]
                thresholds = np.unique(x_column)
                for threshold in thresholds:
                    p = 1
                    predictions = np.ones(n_samples)
                    predictions = np.where(x_column < threshold, -1, 1)

                    misclassified = w[y != predictions]
                    error = sum(misclassified)
                    if error > 0.5:
                        error = 1 - error
                        p = -1

                    if error < min_error:
                        min_error = error
                        clf.polarity = p
                        clf.threshold = threshold
                        clf.feature_idx = feature_i

            EPS = 1e-10
            clf.alpha = 0.5 * np.log((1-error) / (error+EPS))

            predictions = clf.predict(X)

            w *= np.exp(-clf.alpha * y * predictions)
            w /= np.sum(w)

            self.clfs.append(clf)

    def predict(self, X):
        clf_preds = [clf.alpha * clf.predict(X) for clf in self.clfs]
        y_pred = np.sum(clf_preds, axis=0)
        y_pred = np.sign(y_pred)
        return y_pred
