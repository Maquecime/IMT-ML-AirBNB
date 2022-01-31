# -*- coding: utf-8 -*-
"""
Created on Thu Jan 27 11:27:20 2022

@author: samue
"""
import pandas as pd
import re

amenitiesList = ["Internet", "Air conditioning", "Breakfast", "TV", "Bathub", "Dryer", "Elevator in building", "Parking",
                     "Gym", "Heating", "Kitchen", "Pets allowed", "Pool", "Smoking allowed", "Washer", "Wheelchair accessible"]

def getColEncoded(name, dataset):
    return [x for x in dataset.columns if re.search(name + " \d+", str(x))]

def GetDataSet(path):
    dataset = pd.read_csv(path, dtype={'Zipcode': "string"})
    dataset = dataset[dataset["Review Scores Value"].notna()]
    
    neiColEncodedNeighbourhood = getColEncoded("Neighbourhood", dataset)
    neiColEncodedPropertyType = getColEncoded("Property Type", dataset)
    neiColEncodedRoomType = getColEncoded("Room Type", dataset)
    neiColEncodedBedType = getColEncoded("Bed Type", dataset)
    
    X = dataset[["Square Meter"] +
                    ["Accommodates"] +
                    ["Bathrooms"] +
                    ["Bedrooms"] +
                    ["Review Scores Value"] +
                    amenitiesList +
                    neiColEncodedNeighbourhood +
                    neiColEncodedPropertyType +
                    neiColEncodedRoomType +
                    neiColEncodedBedType].values
    
    y = dataset['Price']
    
    return X, y, dataset
    
