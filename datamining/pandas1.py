#!/usr/bin/env python

from IPython.display import display
#import numpy as np
#import matplotlib.pyplot as plt
import pandas as pd
#import mglearn

data = {
    'Name': ["John", "Anna", "Peter", "Linda"],
    'Location': ["New York", "Paris", "Berlin", "London"],
    'Age': [24, 13, 53, 33]
    }

data_pandas = pd.DataFrame(data)
display(data_pandas)
display(data_pandas[data_pandas.Age > 30])
