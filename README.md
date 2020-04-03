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
- Importing the libraries.
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
- Getting the Data
```
train = pd.read_csv(r'C:\Users\Rishabh\Downloads\titanic (1)\train.csv')
test = pd.read_csv(r"C:\Users\Rishabh\Downloads\titanic (1)\test.csv")
```

- Data Analysis
```
sns.distplot(train['Fare'])
plt.show()
```

- Data Preprocessing
```
train.loc[train["Sex"]=="male","Sex"]=0
train.loc[train["Sex"]=="female","Sex"]=1
```
- Imputing Median values
```
train["Fare"] = train["Fare"].fillna(train["Fare"].median())
```
- Creating New Features
```
for i in [train, test]:
        i['Fam_Size'] = np.where((i['SibSp']+i['Parch']) == 0 , 1,
                            np.where((i['SibSp']+i['Parch']) <= 3,2, 3))
        del i['SibSp']
        del i['Parch']
```
- Dropping Irrelevant Features
```
train = train.drop(['Ticket'], axis=1)
test = test.drop(['Ticket'], axis=1)
```
- Then we dropped the columns which do not contribute much to the survival rate of passengers from the train and test data frames.
- For training the classifier, we used four different methods: Logistic Regression, Second degree Logistic Regression, KNN Classifier and the Random Forest Classifier.
- The accuracies of all four processes were calculated and the confusion Matrices were also displayed. The one with the maximum accuracy was selected for predicting the survival of the test set of passengers.
- The predicted data was saved to a csv file for submission.
