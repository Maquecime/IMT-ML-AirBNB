import numpy as np
import pandas as pd
import sklearn as sk
import keras
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from collections import Counter
from keras.utils import to_categorical

# Get and modify dataset
dataset = pd.read_csv("../Dataset/datasetFilter.csv")
dataset = dataset[dataset['Price'].notna()]
data = dataset[["Square Meter", "AmenitiesScore","Latitude", "Longitude"]].values
target = dataset['Price']

# Create features and target for training and testing
x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.33)

# Create model & train it
reg = LinearRegression().fit(x_train, y_train)

train_score = reg.score(x_train,y_train)
test_score = reg.score(x_test,y_test)

print("Training set score: {:.2f} ".format(train_score))
print("Test set score: {:.2f} ".format(test_score))
print(Counter(dataset['Property Type']).keys())

encoded = to_categorical(dataset['Property Type'])
print(encoded)
# invert encoding
inverted = argmax("Boat")
print(inverted)









