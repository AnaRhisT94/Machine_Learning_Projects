# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 06:48:55 2019

@author: Ilan Aizelman
@Summary: Pre-processing techniques of data
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing dataset
dataset = pd.read_csv("Data.csv")
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values
print(X)  
# Dealing with missing data with SKLearn
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values= 'NaN', strategy = 'mean', axis = 0) #ctrl + I for insepection
imputer = imputer.fit(X[:, 1:3]) # Take care of missing values in 2 columns
X[:, 1:3] = imputer.transform(X[:, 1:3])
print(X)

# Categorical Data - We want to use numberes in our data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:,0])
print(X)
# Prevent ML algorithm from thinking a counter is superior
# So we will use Dummy Encoding to transform the counteries to 3 columns
# e.g., France: [1, 0, 0] so Spain and Germany will have zero values
# This is done by OneHotEncoder
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()
print(X)
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)
print(y)

# Splitting the Dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
# The model is built on the traning set only!
# Then, with the built model we check the perf. on test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature scaling
# Age and Salary - are not in the same scale, this will cause issues.
# Why? Because many of the algorithms are based on L2 distance (Euclidean)
# So, computing the L2 dist. between 2 diff. observations, Salary will dominate
# Also in Decision Trees we dont have L2 dist., but it will converge much faster when feat. scaled
# Thus, we can standadisation / normalization
# Standardisation: x_stand = x - x_mean / std(x)
# normalization: x_norm = x - min(x) / max(x) - min(x)
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test) # Here we don't use fit, because we want X_test to be fitted similarly to X_train, we don't need to re-fit again based on X_test
# Do we need to change Dummy Variables? - Not necessarily. It depends on the context
# If want to keep the interp. of the values
# Do we need to apply feat. scaling on y - No, because it's classification with discretised values, it's not regression with continues values.

# Prepartion of Data Preprocessing Template:

