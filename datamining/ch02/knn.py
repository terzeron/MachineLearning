#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import numpy as np
import operator


def create_data_set():
    group = np.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


def classify0(in_x, data_set, labels, k):
    # 거리계산
    #  height
    data_set_size = data_set.shape[0]
    diff_mat = np.tile(in_x, (data_set_size, 1)) - data_set
    sq_diff_mat = diff_mat ** 2
    sq_distances = sq_diff_mat.sum(axis=1)
    distances = sq_distances ** 0.5
    sorted_distIndices = distances.argsort()

    # 가장 짧은 k 거리를 투표
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sorted_distIndices[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1

    # 정렬
    sorted_class_count = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_class_count[0][0]

def file2matrix(filename):
    fr = open(filename)
    numberOfLines = len(fr.readlines())
    returnMat = np.zeros((numberOfLines, 3))
    classLabelVector = []
    fr = open(filename)
    index = 0
    for line in fr.readlines():
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index, :] = listFromLine[0:3]
        classLabelVector.append(listFromLine[-1])
        index += 1
    return returnMat, classLabelVector


def autoNorm(dataSet):
    minVals = dataSet.min(0)
    #print("minVals=", minVals)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    #print("ranges=", ranges)
    normDataSet = np.zeros(np.shape(dataSet))
    m = dataSet.shape[0]
    #print("tile(minVal, (m, 1))=", np.tile(minVals, (m, 1)))
    normDataSet = dataSet - np.tile(minVals, (m, 1))
    #print("normDataSet=", normDataSet)
    #print("tile(ranges, (m, 1))=", np.tile(ranges, (m, 1)))
    newNormDataSet = normDataSet / np.tile(ranges, (m, 1))
    #print("newNormDataSet=", newNormDataSet)
    return newNormDataSet, ranges, minVals


def datingClassTest():
    hoRatio = 0.10
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    print("datingDataMat=", datingDataMat)
    print("datingLables=", datingLabels)
    normMat, ranges, minVals = autoNorm(datingDataMat)
    print("normMat=", normMat)
    print("ranges=", ranges)
    print("minVals=", minVals)
    m = normMat.shape[0]
    numTestVecs = int(m * hoRatio)
    errorCount = 0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i, :], normMat[numTestVecs:m, :], datingLabels[numTestVecs:m], 3)
        print("the classifier came back with %s, the real answer is %s" % (classifierResult, datingLabels[i]))
        if classifierResult != datingLabels[i]:
            errorCount += 1.0
    print("the total error rate is %f" % (errorCount / float(numTestVecs)))

