import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt
from RandomForest import RandomForest
# best plot colours:
# royalblue, lightcoral, salmon, gold, forestgreen, limegreen, mediumseagreen, springgreen, teal, cornflowerblue, navy, darkorchid, purple


def accuracy(y_true, y_predicted):
    acc = np.sum(y_true == y_predicted) / len(y_true)
    return acc


if __name__ == '__main__':
    print("Random Forest")
    # one of the most popular and most powerful
    # train multiple trees into a forest
    # each tree gets random subset of a data (thus random)
    # then with each tree we make a prediction
    # and at the end we do a majority vote (final prediction)

    # by building more trees we have more chances to get correct answer
    # and we don't overfit

    bc = datasets.load_breast_cancer()
    X, y = bc.data, bc.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

    clf = RandomForest(number_trees=3, max_depth=10)
    clf.fit(X_train, y_train)

    y_prediction = clf.predict(X_test)

    print(f'Accuracy: {accuracy(y_test, y_prediction)}')
