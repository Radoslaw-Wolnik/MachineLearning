import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt
from DecisionTree import DecisionTree


def accuracy(y_true, y_predicted):
    acc = np.sum(y_true == y_predicted) / len(y_true)
    return acc

# best plot colours:
# royalblue, lightcoral, salmon, gold, forestgreen, limegreen, mediumseagreen, springgreen, teal, cornflowerblue, navy, darkorchid, purple


if __name__ == '__main__':
    print("Decision tree")

    bc = datasets.load_breast_cancer()
    X, y = bc.data, bc.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

    # fig = plt.figure(figsize=(8, 6))
    # plt.scatter(X[:, 0], y, color='royalblue', marker='o', s=20, alpha=0.5)
    # plt.show()
    clf = DecisionTree(max_depth=10)
    clf.fit(X_train, y_train)

    y_predicted = clf.predict(X_test)
    print(f'Decision tree accuracy: {accuracy(y_test, y_predicted)}')
