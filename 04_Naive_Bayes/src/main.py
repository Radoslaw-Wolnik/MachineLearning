import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt
from NaiveBayes import NaiveBayes
# best plot colours:
# royalblue, lightcoral, salmon, gold, forestgreen, limegreen, mediumseagreen, springgreen, teal, cornflowerblue, navy, darkorchid, purple


def accuracy(y_true, y_predicted):
    acc = np.sum(y_true == y_predicted) / len(y_true)
    return acc


if __name__ == '__main__':
    print("Naive Bayes")

    X, y = datasets.make_classification(n_samples=1000, n_features=10, n_classes= 2, random_state=123)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

    # fig = plt.figure(figsize=(8, 6))
    # plt.scatter(X[:, 0], y, color='royalblue', marker='o', s=20, alpha=0.5)
    # plt.show()

    nb = NaiveBayes()
    nb.fit(X_train, y_train)
    predictions = nb.predict(X_test)

    print(f'Naive Bayes classification accuracy: {accuracy(y_test, predictions)}')
