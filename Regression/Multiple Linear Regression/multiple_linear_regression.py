# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 10:04:36 2019

@author: Ilan Aizelman
@Summary: Multiple Linear Regression
# Tips: Always omit one dummy variable
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing dataset
dataset = pd.read_csv("50_Startups.csv")
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

# Categorical Data - We want to use numberes in our data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 3] = labelencoder_X.fit_transform(X[:,3])
print(X)
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()
print(X)

# Avoiding the Dummy Variable Trap
X = X[:, 1:] # We don't need to do it if the library we use already does that


# Splitting the Dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Fitting Multiple Linear Regression to the Training Set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)

# Building the optimal model using Backward Elimination
import statsmodels.formula.api as sm
# This is that x_0 will be 1, so the coefficiemt b_0 will have x_0 = 1, for our formula
X = np.append(arr = np.ones((50,1)).astype(int), values = X, axis = 1)


# Using just p-values
def backwardElimination(x, sl=0.05):
    """
    Use backward elimination to find the most valuable variables
    Backward Elimination algorithm:
    Step 1 : Select a significance level to stay in the model (e.g. SL = 0.05)
    Sep 2 : Fit the full model with all possible predictors/vars 
    Step 3: look for the highest P-value. if P > SL, goto step 4, otherwise Finish.
    Step 4: remove the variable
    Step 5: Fit model without this variable -> Goto step 3
    x : input data with appended ones in the first column
    sl : significance level, by default 5%
    """
    numVars = len(x[0])
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(y, x).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        if maxVar > sl:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    x = np.delete(x, j, 1)
    regressor_OLS.summary()
    return x
 
SL = 0.05
# The optimal matrix with the ind. vars that have high impact
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
X_Modeled = backwardElimination(X_opt, SL)

