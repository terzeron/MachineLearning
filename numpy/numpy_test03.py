#!/usr/bin/env python

import time
import numpy as np
import pandas as pd

df = pd.DataFrame(np.random.randint(0, 50, size=(50000, 4)), columns=('a', 'b', 'c', 'd'))
df.shape
# (5000000, 5)
df.head()

# iteration

start = time.time()
for idx, row in df.iterrows():
    if row.a == 0:
        df.at[idx, 'e'] = row.d
    elif (row.a <= 25) & (row.a > 0):
        df.at[idx, 'e'] = (row.b) - (row.c)
    else:
        df.at[idx, 'e'] = row.b + row.c
end = time.time()
print(end - start)

# vectorization

start = time.time()
df['e'] = df['b'] + df['c']
df.loc[df['a'] <= 25, 'e'] = df['b'] - df['c']
df.loc[df['a'] == 0, 'e'] = df['d']
end = time.time()
print(end - start)
