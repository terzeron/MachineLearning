#!/usr/bin/env python

# https://m.blog.naver.com/ssdyka/221270345638

from matplotlib import pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import mglearn

# 인위적인 2차원 데이터셋
X, y = make_blobs(random_state=1)
print(X)
print(y)

kmeans = KMeans(n_clusters=3)
kmeans.fit(X)
print("클러스터 레이블:\n{}".format(kmeans.labels_))
print("클러스터 레이블:\n{}".format(kmeans.predict(X)))

fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# 두 개의 클러스터 중심
kmeans = KMeans(n_clusters=2)
kmeans.fit(X)
assignments = kmeans.labels_
mglearn.discrete_scatter(X[:, 0], X[:, 1], assignments, ax=axes[0])

# 다섯개의 클러스터 중심
kmeans = KMeans(n_clusters=5)
kmeans.fit(X)
assignments = kmeans.labels_
mglearn.discrete_scatter(X[:, 0], X[:, 1], assignments, ax=axes[1])

plt.show()