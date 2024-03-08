# some error instead of pip install sklearn use -> pip install scikit-learn || but in import etc use namespace sklearn
# following tutorial https://www.youtube.com/watch?v=ngLyX54e1LU&list=PLqnslRFeH2Upcrywf-u2etjdxxkL8nl7E&index=1&t=21s
# https://www.youtube.com/playlist?list=PLqnslRFeH2Upcrywf-u2etjdxxkL8nl7E
# https://www.youtube.com/playlist?list=PLblh5JKOoLUICTaGLRoHQDuF_7q2GfuJF
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

if __name__ == '__main__':
    cmap = ListedColormap(['#FFO000', '#00FF00', '#0000FF'])
    iris = datasets.load_iris()

    X, y = iris.data, iris.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

    print(X_train.shape)  # we have 120 data vectors with 4 features
    print(X_train[0])  # the first data shows we have vectors with four features: [5.1 2.5 3.  1.1]

    print(y_train.shape)  # we have 120 labels - same number as data
    print(y_train)  # 120 one dimensional vectors
