# -*- coding: utf-8 -*-
"""
Created on Tue Jan 18 08:55:02 2022

@author: lele8
"""
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt

from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.datasets import make_regression
from sklearn.ensemble import GradientBoostingRegressor



datasetTree = pd.read_csv("Dataset/datasetFilter.csv")


amenitiesList = ["Internet", "Air conditioning", "Breakfast", "TV", "Bathub", "Dryer", "Elevator in building", "Parking",
                     "Gym", "Heating", "Kitchen", "Pets allowed", "Pool", "Smoking allowed", "Washer", "Wheelchair accessible"]

def getColEncoded(name):
    return [x for x in newDataset.columns if re.search(name + " \d+", str(x))]


datasetTree = datasetTree[datasetTree['Price'].notna()]
neiColEncodedNeighbourhood = getColEncoded("Neighbourhood")
neiColEncodedPropertyType = getColEncoded("Property Type")
neiColEncodedRoomType = getColEncoded("Room Type")
neiColEncodedBedType = getColEncoded("Bed Type")

### -------- gradient
X = datasetTree[["Square Meter"] +
                ["Accommodates"] +
                ["Bathrooms"] +
                ["Bedrooms"] +²
                ["Guests Included"] +
                neiColEncodedNeighbourhood +
                neiColEncodedPropertyType +
                neiColEncodedRoomType +
                neiColEncodedBedType].values

y = datasetTree["Price"].values

X_train, X_test, y_train, y_test = train_test_split(X, y , test_size=0.33)

reg = GradientBoostingRegressor(random_state=0)
reg.fit(X_train, y_train)
GradientBoostingRegressor(random_state=0)


print("Score: ", reg.score(X_test, y_test))

### -------- tree
"""
X_train, X_test, y_train, y_test = train_test_split(datasetTree[["Square Meter"] + 
                                                                ["Bathrooms"] +
                                                                ["Bedrooms"] +
                                                                ["Guests Included"] +
                                                                neiColEncoded].values
                                                    , datasetTree["Price"].values, 
                                                    test_size=0.33)



#X = datasetTree[["Square Meter", "AmenitiesScore"]].values
#y = datasetTree["Price"].values

clf = tree.DecisionTreeRegressor()

print("Cross Val : " + str(cross_val_score(clf, data, target, cv=10)))

clf = clf.fit(X_train, y_train)
print(clf.predict([X_test[0]]))
print(clf.predict([X_test[1]]))
print(clf.predict([X_test[2]]))

print("Score: ", clf.score(X_test, y_test))
"""
### -------- cross validate tree
"""
clf = tree.DecisionTreeRegressor()

data = datasetTree[["Square Meter"] + 
                ["Bathrooms"] +
                ["Bedrooms"] +
                ["Guests Included"] +
                neiColEncoded].values
target = datasetTree["Price"].values
print("Cross Val : " + str(cross_val_score(clf, data, target, cv=10)))
"""

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
