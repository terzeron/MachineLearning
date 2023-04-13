#!/usr/bin/env python

from sklearn.datasets import load_iris

iris_dataset = load_iris()
print("iris_dataset key: {}".format(iris_dataset.keys()))
# 데이터
#print(iris_dataset['data'])
print(iris_dataset['feature_names'])
# 정답(레이블)
#print(iris_dataset['target'])
print(iris_dataset['target_names'])
#print(iris_dataset['DESCR'])

# 학습 데이터와 테스트 데이터 분리
from sklearn.model_selection import train_test_split
train_input, test_input, train_label, test_label = train_test_split(iris_dataset['data'], iris_dataset['target'], test_size=0.25, random_state=42)
print("shape of train_input: {}".format(train_input.shape))
print("shape of test_input: {}".format(test_input.shape))
print("shape of train_label: {}".format(train_label.shape))
print("shape of test_label: {}".format(test_label.shape))

from sklearn.neighbors import KNeighborsClassifier
# k=3인kNN
#knn = KNeighborsClassifier(n_neighbors=3)
#knn.fit(train_input, train_label)
knn = KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski', metric_params=None, n_jobs=1, n_neighbors=3, p=2, weights='uniform')
knn.fit(train_input, train_label)

import numpy as np
new_input = np.array([[6.1, 2.8, 4.7, 1.2]])
predict_label = knn.predict(new_input)
print(predict_label)
predict_label = knn.predict(test_input)
print(predict_label)
print("test accuracy {:.2f}".format(np.mean(predict_label==test_label)))

