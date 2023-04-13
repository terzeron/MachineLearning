#!/usr/bin/env python

from sklearn.datasets import make_blobs
from scipy.cluster.hierarchy import dendrogram, ward
import matplotlib.pyplot as plt

# 인위적인 2차원 데이터셋
X, y = make_blobs(random_state=0, n_samples=12)

# 데이터 배열 X에 ward함수를 적용
# SciPy의 ward 함수는 병합 군집을 수행할 때 생성된 거리 정보가 담긴 배열을 반환한다.
linkage_array = ward(X)

# 클러스터 간의거리 정보가 담긴 linkage_array를 사용해 덴드로 그램을 그린다.
dendrogram(linkage_array)

# 두개와 세개의 클러스터를 구분하느 커트라인
ax = plt.gca()
bounds = ax.get_xbound()
ax.plot(bounds, [7.25, 7.25], '--', c='k')
ax.plot(bounds, [4, 4], '--', c='k')
ax.text(bounds[1], 7.25, '2 cluster', va='center', fontdict={'size':15})
ax.text(bounds[1], 4, '3 cluster', va='center', fontdict={'size':15})

plt.xlabel("sample num")
plt.ylabel("cluster dist")
plt.show()
