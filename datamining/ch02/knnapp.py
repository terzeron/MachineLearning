#!/usr/bin/env python

import knn
# import matplotlib
import matplotlib.pyplot as plt
import numpy as np

group, labels = knn.create_data_set()
print("group=", group)
print("labels=", labels)

result = knn.classify0([0, 0], group, labels, 3)
print("result=", result)

datingDataMat, datingLabels = knn.file2matrix('datingTestSet.txt')
print("datingDataMat=", datingDataMat)
print("datingLabels=", datingLabels[:20])

'''
fig = plt.figure()
ax = fig.add_subplot(111)
print("datingDataMat[:, 1]=", datingDataMat[:, 1][0])
print("datingDataMat[:, 2]=", datingDataMat[:, 2][0])
ax.scatter(datingDataMat[:, 1], datingDataMat[:, 2])
# ax.scatter(datingDataMat[:, 1], datingDataMat[:, 2], 15.0*array(datingLabels), 15.0*array(datingLabels))
plt.show()
'''

normMat, ranges, minVals = knn.autoNorm(datingDataMat)
print("normMat=", normMat)
print("ranges=", ranges)
print("minVals=", minVals)

knn.datingClassTest()
