#!/usr/bin/env python

import timeit

# timeit을 이용한 반복 작업의 시간 측정

# 1부터 1000까지의 배열에 대해 제곱의 합
print("sum(x * x for x in range(10))=", sum(x * x for x in range(10)))
normal_py_sec = timeit.timeit('sum(x*x for x in range(1000))', number = 10000)

# np.arange()를 이용한 배열의 제곱의 합
import numpy as np
na = np.arange(10)
print("na = np.arange(10); sum(na * na)=", sum(na * na))
naive_np_sec = timeit.timeit('sum(na*na)', setup = "import numpy as np; na = np.arange(1000)", number = 10000)

na = np.arange(10)
print("na = np.arange(10); na.dot(na)=", na.dot(na))
good_np_sec = timeit.timeit('na.dot(na)', setup = "import numpy as np; na = np.arange(1000)", number = 10000)

print("%f sec" % normal_py_sec)
print("%f sec" % naive_np_sec)
print("%f sec" % good_np_sec)

