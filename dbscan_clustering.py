import numpy as np

import matplotlib.pyplot as plt

from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler

centers = [[1, 1], [-1, -1], [1, -1]]
X, labels_true = make_blobs(n_samples=750, centers=centers, cluster_std=0.4,
                            random_state=0)

X = StandardScaler().fit_transform(X)

eps_range = [0.1, 0.125, 0.15, 0.175,
             0.2, 0.225, 0.25, 0.275,
             0.3, 0.325, 0.35, 0.375, 0.4]


plt_x = []
plt_y = []

for e in eps_range:

    db = DBSCAN(eps=e, min_samples=10).fit(X)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    silhouette_avg = metrics.silhouette_score(X, labels)

    print("DBSCAN")
    print("For n_clusters = ", n_clusters_, "/ the silhouette_avg is: ", silhouette_avg)
    print("EPS = ", e)

    plt_y.append(silhouette_avg * 10)
    plt_x.append(e * 30)

plt.plot(plt_x, plt_y, 'ro')
plt.ylabel("Silhouette AVG(times 10)")
plt.xlabel("EPS(times 30)")
plt.axis([0, 13, -5, 10])

plt.show()


