#!/usr/bin/env python

import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
    
def drawScatterChart(x, y): 
    # draw a scatter chart
    plt.scatter(x, y)
    plt.title("Web traffic over the last month")
    plt.xlabel("Time")
    plt.ylabel("Hits/hour")
    plt.xticks([w*7*24 for w in range(10)], ['week %i' % w for w in range(10)])
    plt.autoscale(tight=True)
    plt.grid()

   
data = np.genfromtxt("web_traffic.tsv", delimiter="\t")
print("data=", data)
print("data.shape=", data.shape)
#print(data[:,0])
#print(data[:,1])
x = data[:,0]
y = data[:,1]

print("np.sum(np.isnan(y))=", np.sum(np.isnan(y)))

# extract valid data
x = x[~np.isnan(y)]
y = y[~np.isnan(y)]

drawScatterChart(x, y)

# regression

def error(f, x, y):
    return np.sum((f(x) - y) ** 2)

# linear equation
fp1, residuals, rank, sv, round = np.polyfit(x, y, 1, full=True)
print("Model parameters: %s" % fp1)
print("residuals=%f" % residuals[0])

f1 = np.poly1d(fp1) # regression function
print(error(f1, x, y)) # equal to the residual
fx = np.linspace(0, x[-1], 1000)
plt.plot(fx, f1(fx), linewidth=2)
plt.legend(["f1.order=%i" % f1.order], loc="upper left")

# quadratic equation
f2p = np.polyfit(x, y, 2)
f2 = np.poly1d(f2p)
print(error(f2, x, y))
fx = np.linspace(0, x[-1], 1000)
plt.plot(fx, f2(fx), linewidth=2)
plt.legend(["f2.order=%i" % f2.order], loc="upper left")

plt.show()
