# Kaggle's Titanic Challenge
My analysis for the 'Titanic: Machine Learning from Disaster' competition, hosted by Kaggle.com

## About the Problem
The Titanic Problem is based on the sinking of the ‘Unsinkable’ ship Titanic in the early 1912. It gives us information about numerous passengers like their ages, sexes, sibling counts, embarkment points and whether or not they survived the disaster. Based on these features, you have to predict if an arbitrary passenger on Titanic would survive the sinking.

## Libraries Used
- *Numpy*
- *Pandas*
- *Seaborn*
- *Matplotlib*
- *Sklearn*

## Methodology
- Importing all the necessary libraries.
```
...
import pandas as pd
from sklearn import tree
from sklearn.svm import SVC
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
...
```
- Reading the test and train csv files and loading them to Pandas dataframes

- Checking the density of passengers in certain divisions by plotting its graph vs fare, age, embarked, sex and cabin columns.

- After analyzing the density, dividing the given data such that regions with maximum deviations in densities of passengers can be grouped together for increased accuracy.
- Use numbers in place of alphabets as the algorithms work better with digits. So, we replaced all columns containing alphabets with numbers by assigning each alphabet to a specific number.
- Empty and NaN values were replaced by the median of the remaining values in its column respectively.
- In the ‘Cabin’ column, we extracted first letter of cabin number and replaced them by numbers and empty places were assigned to zero.
- We analyzed the data in the ‘Fare’ and ‘Age’ columns and classified it to produce optimum results.
- We performed all above operations for both train and test.

- Then we dropped the columns which do not contribute much to the survival rate of passengers from the train and test data frames.
- For training the classifier, we used four different methods: Logistic Regression, Second degree Logistic Regression, KNN Classifier and the Random Forest Classifier.
- The accuracies of all four processes were calculated and the confusion Matrices were also displayed. The one with the maximum accuracy was selected for predicting the survival of the test set of passengers.
- The predicted data was saved to a csv file for submission.
