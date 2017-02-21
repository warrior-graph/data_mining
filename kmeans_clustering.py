import matplotlib.pyplot as plt
import time as tm
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_samples, silhouette_score


import numpy as np

x, y = make_blobs(n_samples=1000, centers=5, random_state=1)

range_n_clusters = [x for x in range(3, 30)]


plt_x = []
plt_y = []

for n_clusters in range_n_clusters:
    alg = KMeans(n_clusters=n_clusters, init='k-means++')

    cluster_labels = alg.fit_predict(x)

    silhouette_avg = silhouette_score(x, cluster_labels)
    print("KMeans")
    print("For n_clusters = ", n_clusters, "/ the silhouette_avg is: ", silhouette_avg)

    plt_y.append(silhouette_avg * 100)
    plt_x.append(n_clusters)

plt.plot(plt_x, plt_y, 'ro')
plt.ylabel("Silhouette AVG(times 100)")
plt.xlabel("Number of Clusters")
plt.axis([0, 30, 32, 38])

plt.show()


