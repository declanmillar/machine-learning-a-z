# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Data.csv')
# print(dataset.head)

# select all rows for all columns except the last
# .values returns a NumPy representation of the DataFrame (.to_numpy() is preferred)
X = dataset.iloc[:, :-1].values
# print(X)

# select all the rows for the last column
y = dataset.iloc[:, 3].values
# print(y)

