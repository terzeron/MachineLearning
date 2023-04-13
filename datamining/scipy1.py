#!/usr/bin/env python

import numpy as np
from scipy import sparse

# 대각선 원소가 1이고 나머지는 0인 배열 생성
eye = np.eye(4)
print(eye)

sparse_matrix = sparse.csr_matrix(eye)
print(sparse_matrix)
print("sparse_matrix:{}".format(sparse_matrix))
