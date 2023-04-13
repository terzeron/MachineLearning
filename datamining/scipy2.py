#!/usr/bin/env python

import numpy as np
from scipy import sparse

data = np.ones(4)
print("data=", data)
row_indices = np.arange(4)
print("row_indices=", row_indices)
col_indices = np.arange(4)
print("col_indices=", col_indices)

# coodinate format
eye_coo = sparse.coo_matrix((data, (row_indices, col_indices)))
print("coo format: eye_coo={}".format(eye_coo))
