# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 09:25:03 2019

@author: Ilan Aizelman
@Summary: Support Vector Regression
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing dataset
dataset = pd.read_csv("Position_Salaries.csv")
X = dataset.iloc[:, 1:2].values # Specify it as a matrix
y = dataset.iloc[:, 2:3].values

# Splitting the Dataset into the Training set and Test set
#from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature scaling - Not always used

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y)


# Fitting the SVR Model to the dataset
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(X, y)


# Predicting a new result with Linear Reg
y_pred = regressor.predict(sc_X.transformn(np.array([[6.5]])))


# Visualize SVR. Reg. Results.
plt.scatter(X, y, color = 'red')
plt.plot(X, regressor.predict(X), color = 'blue')
plt.title('Truth or Bluff (SVR)')
plt.xlabel("Position level")
plt.ylabel("Salary")
plt.show()