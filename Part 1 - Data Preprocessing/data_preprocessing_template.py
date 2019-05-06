# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv("Data.csv")
# print(dataset.head)

# select all rows for all columns except the last
# .values returns a NumPy representation of the DataFrame (.to_numpy() is preferred)
X = dataset.iloc[:, :-1].values
# print(X)

# select all the rows for the last column
y = dataset.iloc[:, 3].values
# print(y)

# Taking care of missing data
# In statistics, imputation is the process of replacing missing data with substituted values
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values=np.nan, strategy="mean", axis=0)
imputer = imputer.fit(X[:, 1:3]) # upper bound (3) is not included
X[:, 1:3] = imputer.transform(X[:, 1:3])
# print(X)