#!/usr/bin/env python

import sys
import random
import numpy as np
import pylab as pl
import matplotlib as mpl

def generateNormalRandoms(n):
    return np.random.normal(0, 1, n)

def generateUniformRandoms(n):
    return np.random.uniform(-0.1, 0.1, n)

def generateExponentialRandoms(n):
    return np.random.exponential(1.0, n)

def generatePoissonRandoms(n):
    return np.random.poisson(1.0, n)

def generateParetoRandoms(n):
    return np.random.pareto(3.0, n)

def main():
    n = 10000
    type = "exponential"
    if type == "normal":
        y = generateNormalRandoms(n)
    elif type == "uniform":
        y = generateUniformRandoms(n)
    elif type == "exponential":
        y = generateExponentialRandoms(n)
    elif type == "poisson":
        y = generatePoissonRandoms(n)
    elif type == "pareto":
        y = generateParetoRandoms(n)
    x = []
    for i in range(n):
        x.append(i+1)
    pl.hist(y, bins=int(n/100))
    pl.xlabel('시행')
    pl.ylabel('도수')
    pl.title(type + " distribution")
    pl.show()
    
    
if __name__ == "__main__":
    sys.exit(main())
