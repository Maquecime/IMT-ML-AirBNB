# -*- coding: utf-8 -*-
"""
Created on Mon Jan 17 09:03:58 2022

@author: leo
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv("C:/Users/lele8/OneDrive/Bureau/EMA/Année 2/Machine Learning/Projet/Dataset/datasetFilter.csv", 
                      dtype={'Zipcode': "string", 'License': "string"})


ax = dataset['Neighbourhood'].value_counts().plot(kind='bar',
                                    figsize=(14,8),
                                    title="Distribution of neighbourhood")
ax.set_xlabel("Neighbourhood")
ax.set_ylabel("Frequency")

