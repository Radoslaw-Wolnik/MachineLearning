import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt
from LinearRegression import LinearRegression


# gradient descent method
# look more in depth into it

def mse(y_true, y_predicted):
    # cost function Means Square Error
    return np.mean((y_true - y_predicted) ** 2)


if __name__ == '__main__':
    print("Linear regression")
    # split data to training and samples
    X, y = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=4)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

    fig = plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], y, color='blue', marker='o', s=20)
    plt.show()

    # print(X_train.shape)
    # print(y_train.shape)

    regressor = LinearRegression(learning_rate=0.01)
    regressor.fit(X_train, y_train)
    predicted = regressor.predict(X_test)

    # to measure how our model performance we use Means Square Error
    mse_value = mse(y_test, predicted)
    print(mse_value)

    # plot
    y_predict_line = regressor.predict(X)
    cmap = plt.get_cmap('viridis')
    fig = plt.figure(figsize=(8, 6))
    m1 = plt.scatter(X_train, y_train, color=cmap(0.9), s=10)
    m2 = plt.scatter(X_test, y_test, color=cmap(0.5), s=10)
    plt.plot(X, y_predict_line, color='black', linewidth=2, label="Prediction")
    plt.show()
