# -*- coding: utf-8 -*-
"""
Created on Tue Jan 18 08:55:02 2022

@author: lele8
"""
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import re

from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn import linear_model


datasetTree = pd.read_csv("Dataset/datasetFilter.csv", dtype={'Zipcode': "string"})
datasetTree = datasetTree[datasetTree["Review Scores Value"].notna()]
amenitiesList = ["Internet", "Air conditioning", "Breakfast", "TV", "Bathub", "Dryer", "Elevator in building", "Parking",
                     "Gym", "Heating", "Kitchen", "Pets allowed", "Pool", "Smoking allowed", "Washer", "Wheelchair accessible"]

def getColEncoded(name, dataset):
    return [x for x in dataset.columns if re.search(name + " \d+", str(x))]

def getEcartPrediction(model, X, y):
    ecartPrediction = []
    for i in range(len(y)):
        ecart = abs(model.predict([X[i]]) - y[i])
        ecartPrediction.append(ecart)
    return np.mean(ecartPrediction)
        



neiColEncodedNeighbourhood = getColEncoded("Neighbourhood CLeansed", datasetTree)
neiColEncodedPropertyType = getColEncoded("Property Type", datasetTree)
neiColEncodedRoomType = getColEncoded("Room Type", datasetTree)
neiColEncodedBedType = getColEncoded("Bed Type", datasetTree)


X = datasetTree[["Square Meter"] +
                ["Accommodates"] +
                ["Bathrooms"] +
                ["Bedrooms"] +
                ["Guests Included"] +
                ["Review Scores Value"] +
                amenitiesList +
                neiColEncodedNeighbourhood +
                neiColEncodedPropertyType +
                neiColEncodedRoomType +
                neiColEncodedBedType].values

y = datasetTree["Price"].values

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
rangeCrossValidate = 1

for i in range(rangeCrossValidate):
    
    X_train, X_test, y_train, y_test = train_test_split(X, y , test_size=0.33)
    ## --------- linear regression
    linReg = LinearRegression()
    linReg = linReg.fit(X_train, y_train)
    crossValidateLinearTrain.append(linReg.score(X_train, y_train))
    crossValidateLinearTest.append(linReg.score(X_test, y_test))
    ecartLinear.append(getEcartPrediction(linReg, X_test, y_test))
    ### -------- gradient
    grad = GradientBoostingRegressor()
    grad.fit(X_train, y_train)
    crossValidateGradientTrain.append(grad.score(X_train, y_train))
    crossValidateGradientTest.append(grad.score(X_test, y_test))
    ecartGradient.append(getEcartPrediction(grad, X_test, y_test))

    ### -------- decision tree
    decisionTree = tree.DecisionTreeRegressor()
    decisionTree = decisionTree.fit(X_train, y_train)
    crossValidateTreeTrain.append(decisionTree.score(X_train, y_train))
    crossValidateTreeTest.append(decisionTree.score(X_test, y_test))
    ecartTree.append(getEcartPrediction(decisionTree, X_test, y_test))
    
    ### -------- random forest
    randForest = RandomForestRegressor()
    randForest.fit(X_train, y_train)
    crossValidateForestTrain.append(randForest.score(X_train, y_train))
    crossValidateForestTest.append(randForest.score(X_test, y_test))
    ecartForest.append(getEcartPrediction(randForest, X_test, y_test))
    
    ### -------- bayesia ridge
    bayRidge = linear_model.BayesianRidge()
    bayRidge = bayRidge.fit(X_train, y_train)
    crossValidateBayesianTrain.append(bayRidge.score(X_train, y_train))
    crossValidateBayesianTest.append(bayRidge.score(X_test, y_test))
    ecartBayes.append(getEcartPrediction(bayRidge, X_test, y_test))
    
print("Linear Regression:")    
print("\t Score Train: ", np.mean(crossValidateLinearTrain))
print("\t Score Test: ", np.mean(crossValidateLinearTest))   
print("\t Ecart Mean: ", np.mean(ecartLinear))    
print("Gradient:")    
print("\t Score Train: ", np.mean(crossValidateGradientTrain))
print("\t Score Test: ", np.mean(crossValidateGradientTest))
print("\t Ecart Mean: ", np.mean(ecartGradient))    
print("Decision Tree:")
print("\t Score Train: ", np.mean(crossValidateTreeTrain))
print("\t Score Test: ", np.mean(crossValidateTreeTest))
print("\t Ecart Mean: ", np.mean(ecartTree))    
print("Random Forest:")
print("\t Score Train: ", np.mean(crossValidateForestTrain))
print("\t Score Test: ", np.mean(crossValidateForestTest))
print("\t Ecart Mean: ", np.mean(ecartForest))    
print("Bayesian Ridge:")
print("\t Score Train: ", np.mean(crossValidateBayesianTrain))
print("\t Score Test: ", np.mean(crossValidateBayesianTest))
print("\t Ecart Mean: ", np.mean(ecartBayes))    



####
"""
fn = ["Square Meter", "AmenitiesScore"]
cn= ['Price']

# Setting dpi = 300 to make image clearer than default
fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (4,4), dpi=300)

tree.plot_tree(clf,
           feature_names = fn, 
           class_names=cn,
           filled = True);

#fig.savefig('imagename.png')

#print(tree.plot_tree(clf)[:100])
"""
