#!/usr/bin/env python

from matplotlib import pyplot as plt
from sklearn.datasets import load_iris
import numpy as np

# load data
data = load_iris()
# 데이터 구조는 다음과 같음
'''
150,4,setosa,versicolor,virginica
5.1,3.5,1.4,0.2,0
4.9,3.0,1.4,0.2,0
4.7,3.2,1.3,0.2,0
'''
# 총 150건
features = data['data']
print("features=")
print(features)
feature_names = data['feature_names']
print("feature_names=", feature_names)
target = data['target']
print("target=", target)

# 꽃잎길이
plength = features[:, 2]
target_names = data['target_names']
labels = target_names[target]

is_setosa = (labels == "setosa")

max_setosa = plength[is_setosa].max()
min_non_setosa = plength[~is_setosa].min()
print("Maximum of setosa: {0}.".format(max_setosa))
print("Minimum of others: {0}.".format(min_non_setosa))

# 세토사 구분하기
'''
if features[:, 2] < 2.0:
    print("Iris Setosa")
else:
    print("Iris Virginica or Iris Versicolour")
'''

# 세토사가 아닌 꽃의 구분
features = features[~is_setosa]
labels = labels[~is_setosa]
virginica = (labels == 'virginica')

# 정확도가 가장 높은 경계를 찾기
best_accuracy = -1.0
for fi in range(features.shape[1]):
    # generateo all available boundaries
    thresh = features[:, fi].copy()
    thresh.sort()
    # test all boundaries
    for t in thresh:
        pred = (features[:, fi] > t)
        accuracy = (pred == virginica).mean()
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_fi = fi
            best_t = t
print("Best feature: {0}".format(best_fi))
print("Best threshold: {0}".format(best_t))

# 버지니카와 버시컬러 구분
'''
if example[best_fi] > t:
    print("virginica")
else:
    print("versicolor")
'''


def distance(p0, p1):
    # computes squared euclidean distance
    return np.sum((p0 -p1) ** 2)

def nn_classify(training_set, training_labels, new_example):
    dists = np.array([distance(t, new_example) for t in training_set])
    nearest = dists.argmin()
    return training_labels[nearest]


# get z-score
features -= features.mean(axis = 0)
features /= features.std(axis = 0)

