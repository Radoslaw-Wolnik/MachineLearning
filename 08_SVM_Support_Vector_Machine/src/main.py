import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt
from SVM import SVM
# best plot colours:
# royalblue, lightcoral, salmon, gold, forestgreen, limegreen, mediumseagreen, springgreen, teal, cornflowerblue, navy, darkorchid, purple


def visuals(X, weight, bias):
    def get_hyperplane(x, w, b, offset):
        return (-w[0] * x + b + offset) / w[1]
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    col = np.where(X[:, 0] < 2.5, 'b', 'r')
    plt.scatter(X[:, 0], X[:, 1], marker='o', c=col)
    x0_1 = np.amin(X[:, 0])
    x0_2 = np.amax(X[:, 0])

    x1_1 = get_hyperplane(x0_1, weight, bias, 0)
    x1_2 = get_hyperplane(x0_2, weight, bias, 0)

    x1_1_m = get_hyperplane(x0_1, weight, bias, -1)
    x1_2_m = get_hyperplane(x0_2, weight, bias, -1)

    x1_1_p = get_hyperplane(x0_1, weight, bias, 1)
    x1_2_p = get_hyperplane(x0_2, weight, bias, 1)

    ax.plot([x0_1, x0_2], [x1_1, x1_2], 'y--')
    ax.plot([x0_1, x0_2], [x1_1_m, x1_2_m], 'k')
    ax.plot([x0_1, x0_2], [x1_1_p, x1_2_p], 'k')

    x1_min = np.amin(X[:, 1])
    x1_max = np.amax(X[:, 1])
    ax.set_ylim([x1_min -3, x1_max +3])

    plt.show()


if __name__ == '__main__':
    print("Support Vector Machine")
    # find a hiperplane that best separates our data

    X, y = datasets.make_blobs(n_samples=50, n_features=2, centers=2, cluster_std=1.05, random_state=40)
    y = np.where(y == 0, -1, 1)

    clf = SVM()
    clf.fit(X, y)
    visuals(X, clf.w, clf.b)
