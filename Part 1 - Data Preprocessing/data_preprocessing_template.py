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

# Encoding categorical data
# Encoding is the process of converting data into a format required for a number of information processing needs
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
# print(X)

# Need to prevent the ML equations from assigning significance to the encoded countries, which are categorical variables; use dummy variables; use OneHotEncoder
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()
# print(X)

# Now do the same for y
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)
print(y)
