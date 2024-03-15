import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt
# from matplotlib import colormaps
from PCA import PCA
# best plot colours:
# royalblue, lightcoral, salmon, gold, forestgreen, limegreen, mediumseagreen, springgreen, teal, cornflowerblue, navy, darkorchid, purple


if __name__ == '__main__':
    print("Principal Component Analysis")
    # should be nr 08 - swap with SVM

    iris = datasets.load_iris()
    X, y = iris.data, iris.target

    # fig = plt.figure(figsize=(8, 6))
    # plt.scatter(X[:, 0], y, color='royalblue', marker='o', s=20, alpha=0.5)
    # plt.show()

    # project the data onto 2 primary principal components
    pca = PCA(2)
    pca.fit(X)
    X_projected = pca.transform(X)

    print(f'Shape of X: {X.shape}')
    print(f'Shape of projected X: {X_projected.shape}')

    x1 = X_projected[:, 0]
    x2 = X_projected[:, 1]

    plt.scatter(x1, x2, c=y, edgecolors='none', alpha=0.8, cmap='viridis')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.colorbar()
    plt.show()
