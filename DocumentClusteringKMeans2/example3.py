#!/usr/bin/env python

from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import mglearn
import matplotlib.pyplot as plt
import numpy as np

X, y = make_blobs(random_state=170, n_samples=600)
rng = np.random.RandomState(74)
mglearn.discrete_scatter(X[:, 0], X[:, 1])
plt.show()

# 데이터가 길게 늘어지도록 변경한다.
transformation = rng.normal(size=(2, 2))
X = np.dot(X, transformation)

# 3개의 클러스터로 데이터 kmeans 알고리즘 적용
kmeans = KMeans(n_clusters=3)

kmeans.fit(X)
y_pred = kmeans.predict(X)

# 클러스터 할당과 클러스터 중심 나타내기
mglearn.discrete_scatter(X[:, 0], X[:, 1], kmeans.labels_, markers='o')
mglearn.discrete_scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], [0, 1, 2], markers='^', markeredgewidth=2)

plt.legend(["cluster 0", "cluster 1", "cluster 2"], loc='best')
plt.xlabel("attr 0")
plt.ylabel("attr 1")
plt.show()