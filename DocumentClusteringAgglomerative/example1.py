#!/usr/bin/env python

# https://blog.naver.com/ssdyka/221273356011

from sklearn.datasets import make_blobs
from sklearn.cluster import AgglomerativeClustering
import mglearn
import matplotlib.pyplot as plt

# 인위적인 2차원 데이터셋
X, y = make_blobs(random_state=1)

agg = AgglomerativeClustering(n_clusters=3)
# 새로운 데이터 포인트에 대해서는 예측 불가하여 predict() 제공하지 않음
# 대신 훈련세트로 모델을 만들어서 클러스터 소속정보를 얻기 위해 fit_predict를 사용함
assignment = agg.fit_predict(X)

mglearn.discrete_scatter(X[:, 0], X[:, 1], assignment)
plt.legend(["cluster 0", "cluster 1"], loc="best")
plt.xlabel("attr 0")
plt.ylabel("attr 1")
plt.show()
