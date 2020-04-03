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
• Firstly, we imported all the necessary data analysis, plotting and ML libraries.
• Then we imported the ‘train’ and ‘test’ csv files and assigned them to pandas’ data frames.
• Then we checked the density of passengers in certain divisions by plotting its graph vs fare, age, embarked, sex and cabin columns.
• After analyzing the density, we divided the given data such that regions with maximum deviations in densities of passengers can be grouped together for increased accuracy.
• We use numbers in ML in place of alphabets as the ML algorithms work better with digits. So, we replaced all columns containing alphabets with numbers by assigning each alphabet to a specific number.
• Empty and NaN values were replaced by the median of the remaining values in its column respectively.
• In the ‘Cabin’ column, we extracted first letter of cabin number and replaced them by numbers and empty places were assigned to zero.
• We analyzed the data in the ‘Fare’ and ‘Age’ columns and classified it to produce optimum results.
• We performed all above operations for both train and test.
Object Oriented Programming 3
• Then we dropped the columns which do not contribute much to the survival rate of passengers from the train and test data frames.
• For training the classifier, we used four different methods: Logistic Regression, Second degree Logistic Regression, KNN Classifier and the Random Forest Classifier.
• The accuracies of all four processes were calculated and the one with the maximum accuracy was selected for predicting the survival of the test set of passengers.
• The predicted data was saved to a csv file for submission.