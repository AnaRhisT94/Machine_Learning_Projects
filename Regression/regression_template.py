# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 09:55:16 2019

@author: ii
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing dataset
dataset = pd.read_csv("Position_Salaries.csv")
X = dataset.iloc[:, 1:2].values # Specify it as a matrix
y = dataset.iloc[:, 2].values

# Splitting the Dataset into the Training set and Test set
#from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature scaling - Not always used
'''
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)

'''


# Fitting the Regression Model to the dataset
# Create your regressor here

# Predicting a new result with Linear Reg
y_pred = regressor.predict(6.5)

# Visualize Linear Reg. Results
plt.scatter(X, y, color = 'red')
plt.plot(X, regressor.predict(X), color = 'blue')
plt.title('Truth or Bluff (""Reg Algo Name"")')
plt.xlabel("Position level")
plt.ylabel("Salary")
plt.show()

# Visualize Poly. Reg. Results.
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (""Reg Algo Name"")')
plt.xlabel("Position level")
plt.ylabel("Salary")
plt.show()