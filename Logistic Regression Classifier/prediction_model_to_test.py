# -*- coding: utf-8 -*-
"""
Created on Mon Oct  1 23:47:54 2018

@author: Priyank H
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing the training dataset
dataset = pd.read_csv('train.csv')
x_train = dataset.iloc[:,:-1].values
y_train = dataset.iloc[:, -1].values

#feature scaling of training dataset
from sklearn.preprocessing import StandardScaler
feature_sc_train = StandardScaler()
x_train = feature_sc_train.fit_transform(x_train)

#importing test dataset
test_dataset = pd.read_csv('test.csv')
test_dataset = test_dataset.iloc[:,1:].values

#feature scaling of test dataset
feature_sc_test = StandardScaler()
test_dataset = feature_sc_test.fit_transform(test_dataset)
 
#logistic regression classifier
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(multi_class='multinomial',class_weight='balanced', solver='saga')
lr.fit(x_train,y_train)
y_predict = lr.predict(test_dataset)
 
#adding result to csv file
df = pd.DataFrame(y_predict)
df.columns = ['price_range']
df.index = df.index + 1
df.to_csv('result.csv',encoding='utf-8',index=True)

