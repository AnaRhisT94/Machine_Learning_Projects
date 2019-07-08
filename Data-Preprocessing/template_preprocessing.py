# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 08:35:22 2019

@author: Ilan Aizelman
@Summary: Template for ML algorithms
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# Importing dataset
dataset = pd.read_csv("Data.csv")
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values

# Splitting the Dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature scaling - Not always used
'''
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
'''
