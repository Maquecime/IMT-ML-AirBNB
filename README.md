# IMT-ML-AirBNB

[link to the original dataset](https://public.opendatasoft.com/explore/dataset/airbnb-listings/information/?disjunctive.host_verifications&disjunctive.am&disjunctive.amenities&disjunctive.features).

### IMT Machine Learning Project.

As a Paris based investor that just purchased a property, we want to rent it out on the plateform AirBnb at some point in the year, and we'd like to be able to estimate via a machine learning algorithm, the best price we could set for the rent.

To achieve that, we want to work with a given free to use dataset. We're going to rework it and refactor it, to shape it the way we intend to use it. Indeed, some numeric data are yet not usable in their current format, and needs to be refactored.

This problematic is rooted in the regression problem, with a continuous prediction function.

### Code structure explaination

main.py             -> script to train and test several models using cross validation.  
utils.py            -> useful function for our work.  
Sorting/SortData.py -> script to filter the original dataset to the dataset use to train our models.  
DataSet             -> location of the original dataset and our dataset rework.  
Analyse/Analyse.py  -> script to generate plot for analysing our dataset.  