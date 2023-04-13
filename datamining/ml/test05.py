p#!/usr/bin/env python

from matplotlib import pyplot as plt
from sklearn.datasets import load_iris
import numpy as np

# load data
data = load_iris()
features = data['data']
feature_names = data['feature_names']
target = data['target']

for t, marker, c in zip(range(3), ">ox", "rgb"):
    # 0: 꽃받침길이
    # 1: 꽃받침너비
    # 2: 꽃잎길이
    # 3: 꽃잎너비
    plt.scatter(features[target == t, 0], features[target == t, 2], marker=marker, c=c)
    
plt.show()
