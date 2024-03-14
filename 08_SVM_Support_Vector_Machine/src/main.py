import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt
# from import
# best plot colours:
# royalblue, lightcoral, salmon, gold, forestgreen, limegreen, mediumseagreen, springgreen, teal, cornflowerblue, navy, darkorchid, purple


if __name__ == '__main__':
    print("Project name")

    bc = datasets.load_breast_cancer()
    X, y = bc.data, bc.target

    fig = plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], y, color='royalblue', marker='o', s=20, alpha=0.5)
    plt.show()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)
