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

from sklearn.cluster import KMeans
#k_means = KMeans(n_clusters=3)
k_means = KMeans(algorithm='auto', copy_x=True, init='k-means++', max_iter=300, n_clusters=3, n_init=10, random_state=None, tol=0.0001, verbose=0)
k_means.fit(train_input)
# k=3이므로 0,1,2번 클러스터가 생성됨
# 0번 클러스터는 주로 2번과 1번 품종의 데이터가 모여있음
print("0 cluster:", train_label[k_means.labels_==0])
# 1번 클러스터는 0번 품종의 데이터가 모여있음
print("1 cluster:", train_label[k_means.labels_==1])
# 2번 클러스터는 ᆻ주로 2번 품종의 데이터가 모여있음
print("2 cluster:", train_label[k_means.labels_==2])

import numpy as np
new_input = np.array([[6.1, 2.8, 4.7, 1.2]])
prediction = k_means.predict(new_input)
print(prediction)
predict_cluster = k_means.predict(test_input)
print(predict_cluster)

# 클러스터 번호 -> 품종 번호
# 매번 랜덤하게 나옴
np_arr = np.array(predict_cluster)
np_arr[np_arr==0], np_arr[np_arr==1], np_arr[np_arr==2] = 3, 4, 5
np_arr[np_arr==3] = 1
np_arr[np_arr==4] = 0
np_arr[np_arr==5] = 2
predict_label = np_arr.tolist()
print(predict_label)
print("test accuracy {:.2f}".format(np.mean(predict_label==test_label)))

