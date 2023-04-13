#!/usr/bin/env python

from numpy import *

print(random.rand(4,4))
randMat = mat(random.rand(4, 4))
print(randMat)
invRandMat = randMat.I
print(invRandMat)
print(invRandMat * randMat)
print(invRandMat * randMat - eye(4))
