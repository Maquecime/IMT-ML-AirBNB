import numpy as np
import sklearn as sk
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
import sys
sys.path.insert(0, '..')
from utils import GetDataSet

def getColEncoded(name, dataset):
    return [x for x in dataset.columns if re.search(name + " \d+", str(x))]

X, Y = GetDataSet("../Dataset/datasetFilter.csv")

# Create features and target for training and testing
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

# Create model & train it
reg = LinearRegression().fit(x_train, y_train)

train_score = reg.score(x_train,y_train)
test_score = reg.score(x_test,y_test)

print("Training set score: {:.2f} ".format(train_score))
print("Test set score: {:.2f} ".format(test_score))

# Cross validation
scores = cross_val_score(reg, X, Y, cv=10)

print("Cross validation mean score : {:.2f}".format(scores.mean()))










