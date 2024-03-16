
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt
from LDA import LDA
# best plot colours:
# royalblue, lightcoral, salmon, gold, forestgreen, limegreen, mediumseagreen, springgreen, teal, cornflowerblue, navy, darkorchid, purple


if __name__ == '__main__':
    print("Linear Discriminant Analysis")
    # reducing dimensions of our data without losing too much data
    # better than LDA because we are maximizing separation based on classes - labels
    # supervised

    iris = datasets.load_iris()
    X, y = iris.data, iris.target

    lda = LDA(2)
    lda.fit(X, y)
    x_projected = lda.transform(X)

    print(f'Shape of X: {X.shape}')
    print(f'Shape of projected X: {x_projected.shape}')

    x1 = x_projected[:, 0]
    x2 = x_projected[:, 1]

    plt.scatter(x1, x2, c=y, edgecolors='none', alpha=0.8, cmap='viridis')
    plt.xlabel('Linear Discriminant 1')
    plt.ylabel('Linear Discriminant 2')
    plt.colorbar()
    plt.show()

