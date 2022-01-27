import numpy as np
import pandas as pd
import sklearn as sk
import re
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

def getColEncoded(name, dataset):
    return [x for x in dataset.columns if re.search(name + " \d+", str(x))]

# Get dataset column
amenitiesList = ["Internet", "Air conditioning", "Breakfast", "TV", "Bathub", "Dryer", "Elevator in building", "Parking",
                     "Gym", "Heating", "Kitchen", "Pets allowed", "Pool", "Smoking allowed", "Washer", "Wheelchair accessible"]

# Get and modify dataset
dataset = pd.read_csv("../Dataset/datasetFilter.csv")

neiColEncodedNeighbourhood = getColEncoded("Neighbourhood", dataset)
neiColEncodedPropertyType = getColEncoded("Property Type", dataset)
neiColEncodedRoomType = getColEncoded("Room Type", dataset)
neiColEncodedBedType = getColEncoded("Bed Type", dataset)

data = dataset[["Square Meter"] +
                ["Accommodates"] +
                ["Bathrooms"] +
                ["Bedrooms"] +
                ["Guests Included"] +
                amenitiesList +
                neiColEncodedNeighbourhood +
                neiColEncodedPropertyType +
                neiColEncodedRoomType +
                neiColEncodedBedType].values

target = dataset['Price']

# Create features and target for training and testing
x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.2)

# Create model & train it
reg = LinearRegression().fit(x_train, y_train)

train_score = reg.score(x_train,y_train)
test_score = reg.score(x_test,y_test)

print("Training set score: {:.2f} ".format(train_score))
print("Test set score: {:.2f} ".format(test_score))

# Cross validation
scores = cross_val_score(reg, data, target, cv=10)

print("Cross validation mean score : {:.2f}".format(scores.mean()))










