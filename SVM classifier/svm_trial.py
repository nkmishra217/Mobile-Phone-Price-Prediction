# -*- coding: utf-8 -*-
"""
Created on Wed Oct  3 16:06:05 2018

@author: Priyank H
"""

import numpy as np
import pandas as pd

#importing the dataset
dataset = pd.read_csv('train.csv')
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:, -1].values

#splitting the dataset inot train and test set
from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.15)

#feature scaling
from sklearn.preprocessing import StandardScaler
feature_sc = StandardScaler()
x_train = feature_sc.fit_transform(x_train)
x_test = feature_sc.transform(x_test)

#SVM classifier
from sklearn.svm import SVC
#to get the accuracy score
from sklearn.metrics import accuracy_score
svm = SVC(class_weight='balanced', decision_function_shape='ovr',kernel='linear')
svm.fit(x_train,y_train)
y_predict = svm.predict(x_test)
w = accuracy_score(y_test, y_predict)
print(w)
