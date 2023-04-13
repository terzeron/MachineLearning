#!/usr/bin/env python

import numpy as np

print(np.zeros((3, 4)))

print(np.ones((2, 3, 4), dtype=np.int16))

print(np.empty((2, 3)))

# 10에서 시작해서 30까지 5씩 
print(np.arange(10, 30, 5))

print(np.arange(0, 2, 0.3))

from numpy import pi

# 0~2 사이의 9개 숫자
print(np.linspace(0, 2, 9))

# 180도까지의 100개 구간
x = np.linspace(0, 2*pi, 100)
print(x)
f = np.sin(x)
print(f)
