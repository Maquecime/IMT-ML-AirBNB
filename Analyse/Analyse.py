# -*- coding: utf-8 -*-
"""
Created on Mon Jan 17 09:03:58 2022

@author: leo
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv("../Dataset/datasetFilter.csv", 
                      dtype={'Zipcode': "string", 'License': "string"})

amenitiesList = ["Internet", "Air conditioning", "Breakfast", "TV", "Bathub", "Dryer", "Elevator in building", "Parking",
                     "Gym", "Heating", "Kitchen", "Pets allowed", "Pool", "Smoking allowed", "Washer", "Wheelchair accessible"]
                 
"""
print()

"""
ax = dataset[amenitiesList].sum().sort_values(ascending=False).plot(kind='bar',
                                    figsize=(14,8),
                                    title="Distribution of Amenities")
"""
ax = dataset["Bed Type"].value_counts().plot(kind='bar',
                                    figsize=(14,8),
                                    title="Distribution of Bed Type")

ax.set_xlabel("Bed Type")
ax.set_ylabel("Frequencies")

"""
print(dataset)