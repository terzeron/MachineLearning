#!/usr/bin/env python

import numpy as np

print("numpy.version=", np.__version__)
print("np.dot=", np.dot)

a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])
print(a @ b)       # preferred way
print(np.dot(a, b)) # legacy but still valid

