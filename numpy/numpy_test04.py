#!/usr/bin/env python

import time
import numpy as np

m = np.random.rand(1, 5)
x = np.random.rand(500000, 5)

# iteration

total = 0
tic = time.process_time()
for i in range(0, 500000):
    total = 0
    for j in range(0, 5):
        total = total + x[i][j] * m[0][j]
    #zero[i] = total
toc = time.process_time()
print("Computation time = " + str((toc - tic)) + "seconds")

# vectorization

tic = time.process_time()
# dot product
np.dot(x, m.T)
toc = time.process_time()
print("Computation time = " + str((toc - tic)) + "seconds")
