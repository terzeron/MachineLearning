#!/usr/bin/env python

import time

# iteration

start = time.time()
total = 0
for item in range(0, 15000000):
    total += item
print("sum is: ", total)
end = time.time()
print(end - start)

# verctorization

import numpy as np
start = time.time()
print("sum is: ", np.sum(np.arange(15000000)))
end = time.time()
print(end - start)

