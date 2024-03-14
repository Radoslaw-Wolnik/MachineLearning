import numpy as np

class LogisticRegression:
    # insted of linear output we are guessing the propability
    # using approximation  : ~y = h0(x) = 1 / (1 + e **(-wx + b))
    # and sigmoid function : s(x) = 1 / (1 + e **-x)
    # Cost function - cross entropy
    # Gradient descend - to update rules that our approximation uses
    # learning rate - how far we go with each step

    def __init__(self, learning_rate=0.001, number_iterations=1000):
        self.lr = learning_rate
        self.n_iters = number_iterations
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        # initial parameters
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        # gradient descent
        for _ in range(self.n_iters):
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = self._sigmoid(linear_model)

            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted = self._sigmoid(linear_model)
        y_predicted_class = [1 if data > 0.5 else 0 for data in y_predicted]
        return y_predicted_class

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
