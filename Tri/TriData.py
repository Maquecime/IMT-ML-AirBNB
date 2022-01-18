# -*- coding: utf-8 -*-
"""
Created on Thu Jan 13 10:15:21 2022

@author: leo
"""

import numpy as np
import pandas as pd
import re
import math


pd.set_option('display.max_columns', None)



def initDataset():
    dataset = pd.read_csv("C:/Users/lele8/OneDrive/Bureau/EMA/Année 2/Machine Learning/Projet/Dataset/airbnb-listings.csv", sep=";", 
                          dtype={'Zipcode': "string", 'License': "string"})
    
    usefullColumns = ["ID", "Listing Url", "Name", "Summary", "Space", "Description", "Neighborhood Overview",
                      "Notes", "Access", "Interaction", "Street", "Neighbourhood",
                      "Neighbourhood Cleansed", "Neighbourhood Group Cleansed", "Zipcode",
                      "Latitude", "Longitude", "Property Type", "Room Type", "Accommodates",
                      "Bathrooms", "Bedrooms", "Bed Type", "Amenities", "Square Feet", "Price", "Weekly Price",
                      "Monthly Price", "Cleaning Fee", "Guests Included", "Number of Reviews", "Review Scores Rating",
                      "Review Scores Accuracy", "Review Scores Cleanliness", "Review Scores Checkin", 
                      "Review Scores Communication", "Review Scores Location", "Review Scores Value"
                      ]
    dataset = dataset[usefullColumns]
    dataset = dataset.rename(columns={'Square Feet': 'Square Meter'})
    
    dataset["Square Meter"] = 0
    return dataset
    
def extractNumber(value):
    return float(re.search("^(\d+[\.,]\d|\d+)", value.group(0)).group(0).replace(",", "."))

def calculSquareMeter(dataset):
    
    columnForSquareMeter = ["Name", "Summary", "Description", "Space"]
    formatSquareMeter = ["(\d+[\.,]\d|\d+) *[mM][2²]\D", "(\d+[\.,]\d|\d+) mètres carrés", "(\d+[\.,]\d|\d+) metres carres", "(\d+[\.,]\d|\d+) square meter", 
                         "(\d+[\.,]\d|\d+) square meter", "(\d+[\.,]\d|\d+) mètres carré", "(\d+[\.,]\d|\d+) metres carre",  "(\d+[\.,]\d|\d+)+mcarré", 
                         "(\d+[\.,]\d|\d+) meter surface square", "(\d+[\.,]\d|\d+) *sqm", "(\d+[\.,]\d|\d+)+m carre", "(\d+[\.,]\d|\d+)+m carré",
                         "(\d+[\.,]\d|\d+)+m carrés", "(\d+[\.,]\d|\d+) sq mt", "(\d+[\.,]\d|\d+) sq.mt", "(\d+[\.,]\d|\d+) *m^2(\d+[\.,]\d|\d+)", "(\d+[\.,]\d|\d+) *sq m",  "(\d+[\.,]\d|\d+) meter sq",
                         "(\d+[\.,]\d|\d+) mq", "(\d+[\.,]\d|\d+) sq. meter", "(\d+[\.,]\d|\d+) sq. metre", "(\d+[\.,]\d|\d+) *m sq", "(\d+[\.,]\d|\d+) *sq. mtrs "]
    formatFeetMeter = ["(\d+[\.,]\d|\d+) *(ft|FT)[2²]\D",
                       "(\d+[\.,]\d|\d+) square foot", "(\d+[\.,]\d|\d+) square feet", "(\d+[\.,]\d|\d+) foot  square", "(\d+[\.,]\d|\d+) feet square",
                       "(\d+[\.,]\d|\d+) SQUARE FOOT", "(\d+[\.,]\d|\d+) SQUARE FEET", "(\d+[\.,]\d|\d+) FOOT SQUARE", "(\d+[\.,]\d|\d+) FEET SQUARE",
                       "(\d+[\.,]\d|\d+) sq ft2\D", "(\d+[\.,]\d|\d+) Sq. Ft.", "(\d+[\.,]\d|\d+)+sqft", "(\d+[\.,]\d|\d+) *sqft", "(\d+[\.,]\d|\d+) *sf[ .]"]
    for index, row in dataset.iterrows():
        #print(row["Summary"])
        squareMeter = 0
        finded = False
        #print(index)
        for col in columnForSquareMeter:
            for formatSM in formatSquareMeter:
                imtag = re.search(formatSM, str(row[col]))
                if imtag:      
                    squareMeter = extractNumber(imtag)
                    dataset.loc[index, "Square Meter"] = squareMeter
                    finded = True
                    break
            if(finded):
                break
            else:
                for formatFM in formatFeetMeter:
                    imtag = re.search(formatFM, str(row[col]))
                    if imtag:
                        squareMeter = extractNumber(imtag)
                        dataset.loc[index, "Square Meter"] = squareMeter / 10.7639
                        finded = True
                        break
                    
    return dataset

def calculAmenitiesScore(dataset):
    amenitiesScore = {"Internet" : 10,
                  "Air conditioning" : 6,
                  "Breakfast" : 3,
                  "TV": 5,
                  "Bathub": 1,
                  "Dryer": 3,
                  "Elevator in building" : 1,
                  "Parking" : 8,
                  "Gym" : 3,
                  "Heating" : 7,
                  "Kitchen" : 8,
                  "Pets allowed" : 5,
                  "Pool" : 15,
                  "Smoking allowed" : 2,
                  "Washer" : 4,
                  "Wheelchair accessible" : 3}
    
    newDataset["AmenitiesScore"] = 0
    for index, row in newDataset.iterrows():
        tmpAmenitiesScore = 0
        for i in amenitiesScore:
            if i in str(row["Amenities"]):
                
                tmpAmenitiesScore += amenitiesScore[i]
        newDataset.loc[index, "AmenitiesScore"] = tmpAmenitiesScore
        
    return newDataset

def main():
    dataset = initDataset()
    dataset = calculSquareMeter(dataset)
    newDataset = dataset[(dataset["Square Meter"] != 0) & 
                         (dataset["Square Meter"] > 10) & 
                         (dataset["Square Meter"] < 900)].copy()
    newDataset = calculAmenitiesScore(dataset)
    newDataset.to_csv(path_or_buf="C:/Users/lele8/OneDrive/Bureau/EMA/Année 2/Machine Learning/Projet/Dataset/datasetFilter.csv")

main()







#print(tmpDataset.loc[tmpDataset["ID"] == 25819, "Description"].values)
