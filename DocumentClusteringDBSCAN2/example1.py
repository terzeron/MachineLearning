#!/usr/bin/env python

# https://blog.naver.com/ssdyka/221273386455

from sklearn.datasets import make_blobs
from sklearn.cluster import DBSCAN

# 인위적인 2차원 데이터셋
X, y = make_blobs(random_state=0, n_samples=12)

dbscan = DBSCAN()
clusters = dbscan.fit_predict(X)

print("클러스터 레이블 : \n{}".format(clusters))
