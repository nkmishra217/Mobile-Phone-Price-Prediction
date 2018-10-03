# -*- coding: utf-8 -*-
"""
Created on Sun Sep 30 18:36:47 2018

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
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.10)

#feature scaling
from sklearn.preprocessing import StandardScaler
feature_sc = StandardScaler()
x_train = feature_sc.fit_transform(x_train)
x_test = feature_sc.transform(x_test)

#logistic regression classifier
from sklearn.linear_model import LogisticRegression
#to get the accuracy score
from sklearn.metrics import accuracy_score
lr = LogisticRegression(multi_class='multinomial',class_weight='balanced', solver='saga')
lr.fit(x_train,y_train)
y_predict = lr.predict(x_test)
w = accuracy_score(y_predict,y_test)
print(w)
print(y_predict)

