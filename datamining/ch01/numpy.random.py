#!/usr/bin/env python

from numpy import bmat, random, eye

print(random.rand(4,4))
randMat = bmat(random.rand(4, 4))
print(randMat)
invRandMat = randMat.I
print(invRandMat)
print(invRandMat * randMat)
print(invRandMat * randMat - eye(4))
