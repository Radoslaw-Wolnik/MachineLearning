import numpy as np


class LinearRegression:

    def __init__(self, learning_rate=0.001, n_iterations=1000):
        self.lr = learning_rate
        self.n_iters = n_iterations
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        # initial parameters
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features) # for each component we put zero
        self.bias = 0
        # gradient descent method
        # look into that more
        for _ in range(self.n_iters):
            # approximation of y = weigts * x + bias
            y_predicted = np.dot(X, self.weights) + self.bias
            # derivative of weight
            dw = (1/n_samples) * np.dot(X.T, (y_predicted - y))
            # derivative of bias
            db = (1/n_samples) * np.sum(y_predicted - y)
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        y_predicted = np.dot(X, self.weights) + self.bias
        return y_predicted
