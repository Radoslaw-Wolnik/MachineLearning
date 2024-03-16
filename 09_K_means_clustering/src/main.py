import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
from kMeansCluster import kMeansCluster

# best plot colours:
# royalblue, lightcoral, salmon, gold, forestgreen, limegreen, mediumseagreen, springgreen, teal, cornflowerblue, navy, darkorchid, purple


if __name__ == '__main__':
    print("k Means Clustering")

    X, y = datasets.make_blobs(centers=4, n_samples=500, n_features=2, shuffle=True, random_state=123)
    clusters = len(np.unique(y))

    kMeans = kMeansCluster(K=clusters, max_iterations=150, plot_steps=True)
    y_pred = kMeans.predict(X)

    # kMeans.plot()
