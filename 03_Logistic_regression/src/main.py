import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt
from LogisticRegression import LogisticRegression


def accuracy(y_true, y_predicted):
    acc = np.sum(y_true == y_predicted) / len(y_true)
    return acc


if __name__ == '__main__':
    print("Logistic regression")

    bc = datasets.load_breast_cancer()
    X, y = bc.data, bc.target

    fig = plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], y, color='blue', marker='o', s=20, alpha=0.5)
    plt.show()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

    regressor = LogisticRegression(learning_rate=0.0001, number_iterations=1000)
    regressor.fit(X_train, y_train)
    predictions = regressor.predict(X_test)

    print(f"Logistic regression accuracy: {accuracy(y_test, predictions)}")
    plt.scatter(X_test[:, 0], y_test, color='blue', marker='o', s=20, alpha=0.5)
    plt.scatter(X_test[:, 0], predictions, color='red', marker='o', s=20, alpha=0.2)
    plt.show()
