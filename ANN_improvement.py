#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 30 14:23:15 2021

@author: sethji
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf

# Importing the dataset
dataset = pd.read_csv('DataSet.csv')

#we have to include 3:12 index
X = dataset.iloc[:, 3:13].values
Y = dataset.iloc[:, 13].values

# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder
labelencoder_X2=LabelEncoder()
X[:,2]=labelencoder_X2.fit_transform(X[:,2])
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
X=np.array(ct.fit_transform(X))
X=X[:,1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=0)

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)

classifier=tf.keras.models.Sequential()
classifier.add(tf.keras.layers.Dense(units=6, activation = 'relu'))
classifier.add(tf.keras.layers.Dropout(p=0.1))
classifier.add(tf.keras.layers.Dense(units = 6 ,activation = 'relu'))
classifier.add(tf.keras.layers.Dropout(p=0.1))
classifier.add(tf.keras.layers.Dense(units = 1 , activation = 'sigmoid'))
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
classifier.fit(X_train,Y_train,batch_size=10,epochs=100)

#evaluating the ANN
import keras
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
def build_classifier(optimizer):
    classifier=tf.keras.models.Sequential()
    classifier.add(tf.keras.layers.Dense(units=6, activation = 'relu'))
    classifier.add(tf.keras.layers.Dense(units = 6 ,activation = 'relu'))
    classifier.add(tf.keras.layers.Dense(units = 1 , activation = 'sigmoid'))
    classifier.compile(optimizer=optimizer,loss='binary_crossentropy',metrics=['accuracy'])
    return classifier
#improvement Tuning
classifier = KerasClassifier(build_fn = build_classifier)
parameters = {'batch_size':[25,32],
              'epochs': [100,400],
              'optimizer': ['adam','rmsprop']}
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10,
                           n_jobs=-1)
grid_search = grid_search.fit(X_train, Y_train)
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_
