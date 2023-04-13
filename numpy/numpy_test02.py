#!/usr/bin/env python

import numpy as np
import pandas as pd
df = pd.DataFrame(np.random.randint(0, 50, size=(50000, 4)), columns=('a','b','c','d'))
df.shape
# (5000000, 5)
df.head()

# iteration

import time
start = time.time()
for idx, row in df.iterrows():
    df.at[idx,'ratio'] = 100 * (row["d"] / row["c"])
end = time.time()
print(end - start)

# vectorization

start = time.time()
df["ratio"] = 100 * (df["d"] / df["c"])
end = time.time()
print(end - start)