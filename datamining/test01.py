#!/usr/bin/env python

#from IPython.core import display
#import numpy as np
#import matplotlib.pyplot as plt
#import pandas as pd
#import mglearn
from sklearn.datasets import load_iris

iris_dataset = load_iris()
#print(iris_dataset)
print("keys of iris dataset:", iris_dataset.keys())
print(iris_dataset['DESCR'])
print("target:", iris_dataset['target'])
print("target_names:", iris_dataset['target_names'])
print("shape of target:", iris_dataset['target'].shape)
print("first 5 records of data:", iris_dataset['data'][:5])
print(type(iris_dataset['data']))
print("shape of data:", iris_dataset['data'].shape)
print("feature_names:", iris_dataset['feature_names'])

# 데이터셋을 분리
# 1) 훈련 데이터
#    모델을 만들 때 사용하는 데이터 
# 2) 테스트 데이터(홀드아웃 데이터)
#    모델이 얼마나 잘 동작하는지 측정하는 데이터

# 데이터셋을 3:1의 비율로 분리해주는 유틸리티
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(iris_dataset['data'], iris_dataset['target'], random_state=0)
print("X_train:", X_train.shape)
print("X_train=", X_train)
print("X_test:", X_test.shape)
print("X_test=", X_test)
print("y_train:", y_train.shape)
print("y_train=", y_train)
print("y_test:", y_test.shape)
print("y_test=", y_test)





