#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

# -10부터 10까지 100개의 간격 배열
x = np.linspace(-10, 10, 100)
print("x=", x)
y = np.sin(x)
plt.plot(x, y, marker="x")
plt.show()
