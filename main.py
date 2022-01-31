# -*- coding: utf-8 -*-
"""
Created on Tue Jan 18 08:55:02 2022

@author: lele8
"""
import numpy as np
import math
import matplotlib.pyplot as plt

from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn import linear_model

import sys
sys.path.insert(0,".")
from utils import GetDataSet

X, y, dataset = GetDataSet("Dataset/datasetFilter.csv")

# Variables declaration
crossValidateLinearTrain = []
crossValidateLinearTest = []
ecartLinear = []
crossValidateGradientTrain = []
crossValidateGradientTest = []
ecartGradient = []
crossValidateTreeTrain = []
crossValidateTreeTest = []
ecartTree = []
crossValidateForestTrain = []
crossValidateForestTest= []
ecartForest = []
crossValidateBayesianTrain = []
crossValidateBayesianTest= []
ecartBayes = []
rangeCrossValidate = 10

# Training and testing several models using cross validation
for i in range(rangeCrossValidate):
    X_train, X_test, y_train, y_test = train_test_split(X, y , test_size=0.33)
    ## --------- linear regression
    linReg = LinearRegression()
    linReg.fit(X_train, y_train)
    crossValidateLinearTrain.append(linReg.score(X_train, y_train))
    crossValidateLinearTest.append(linReg.score(X_test, y_test))

    ### -------- gradient
    grad = GradientBoostingRegressor()
    grad.fit(X_train, y_train)
    crossValidateGradientTrain.append(grad.score(X_train, y_train))
    crossValidateGradientTest.append(grad.score(X_test, y_test))
    
    
    ### -------- bayesian ridge
    bayRidge = linear_model.BayesianRidge()
    bayRidge.fit(X_train, y_train)
    crossValidateBayesianTrain.append(bayRidge.score(X_train, y_train))
    crossValidateBayesianTest.append(bayRidge.score(X_test, y_test))
    
    
    X_train, X_test, y_train, y_test = train_test_split(X, y , test_size=0.2)
    ### -------- decision tree
    decisionTree = tree.DecisionTreeRegressor()
    decisionTree.fit(X_train, y_train)
    crossValidateTreeTrain.append(decisionTree.score(X_train, y_train))
    crossValidateTreeTest.append(decisionTree.score(X_test, y_test))

    ### -------- random forest
    randForest = RandomForestRegressor()
    randForest.fit(X_train, y_train)
    crossValidateForestTrain.append(randForest.score(X_train, y_train))
    crossValidateForestTest.append(randForest.score(X_test, y_test))

# Display results 
print("Linear Regression:")    
print("\t Score Train: ", np.mean(crossValidateLinearTrain))
print("\t Score Test: ", np.mean(crossValidateLinearTest))       
print("Gradient:")    
print("\t Score Train: ", np.mean(crossValidateGradientTrain))
print("\t Score Test: ", np.mean(crossValidateGradientTest)) 
print("Bayesian Ridge:")
print("\t Score Train: ", np.mean(crossValidateBayesianTrain))
print("\t Score Test: ", np.mean(crossValidateBayesianTest))     
print("Decision Tree:")
print("\t Score Train: ", np.mean(crossValidateTreeTrain))
print("\t Score Test: ", np.mean(crossValidateTreeTest))
print("Random Forest:")
print("\t Score Train: ", np.mean(crossValidateForestTrain))
print("\t Score Test: ", np.mean(crossValidateForestTest))    
