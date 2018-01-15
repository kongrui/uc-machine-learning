#!/usr/bin/env python

import numpy as np
import pandas as pd
from time import time
from sklearn.metrics import f1_score
import pytablewriter

# Read student data
student_data = pd.read_csv("student-data.csv")
print "Student data read successfully!"

# OBSERVATION -
#  Out of those 395 students, 265 passed and 130 failed.
#  So label data is unbalanced

feature_cols = list(student_data.columns[:-1])
# one series
target_col = student_data.columns[-1]

# Separate the data into feature data and target data (X_all and y_all, respectively)
# feature dataframe and target dataframe
X_all_raw = student_data[feature_cols]
y_all_raw = student_data[target_col]

def preprocess_features(X):
    ''' Preprocesses the student data and converts non-numeric binary variables into
        binary (0/1) variables. Converts categorical variables into dummy variables. '''

    # Initialize new output DataFrame
    output = pd.DataFrame(index=X.index)

    # Investigate each feature column for the data
    for col, col_data in X.iteritems():

        # If data type is non-numeric, replace all yes/no values with 1/0
        if col_data.dtype == object:
            col_data = col_data.replace(['yes', 'no'], [1, 0])

        # If data type is categorical, convert to dummy variables
        if col_data.dtype == object:
            # Example: 'school' => 'school_GP' and 'school_MS'
            col_data = pd.get_dummies(col_data, prefix=col)

            # Collect the revised columns
        output = output.join(col_data)

    return output


X_all = preprocess_features(X_all_raw)
print "Processed feature columns ({} total features):\n{}".format(len(X_all.columns), list(X_all.columns))
y_all = y_all_raw

from sklearn.cross_validation import train_test_split
num_train = 300  # about 75% of the data : 395
num_test = X_all.shape[0] - num_train
X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=num_test)

# Show the results of the split
print "Training set has {} samples.".format(X_train.shape[0])
print "Testing set has {} samples.".format(X_test.shape[0])

def train_classifier(clf, X_train, y_train):
    ''' Fits a classifier to the training data. '''

    # Start the clock, train the classifier, then stop the clock
    start = time()
    clf.fit(X_train, y_train)
    end = time()

    # Print the results
    print "Trained model in {:.4f} seconds".format(end - start)

    return end - start


def predict_labels(clf, features, target):
    ''' Makes predictions using a fit classifier based on F1 score. '''

    # Start the clock, make predictions, then stop the clock
    start = time()
    y_pred = clf.predict(features)
    end = time()

    # Print and return results
    print "Made predictions in {:.4f} seconds.".format(end - start)
    return f1_score(target.values, y_pred, pos_label='yes'), end - start


def train_predict(clf, X_train, y_train, X_test, y_test):
    ''' Train and predict using a classifer based on F1 score. '''

    # Indicate the classifier and the training set size
    print "Training a {} using a training set size of {}. . .".format(clf.__class__.__name__, len(X_train))

    # Train the classifier
    train_classifier(clf, X_train, y_train)

    # Print the results of prediction for both training and testing
    print "F1 score for training set: {:.4f}.".format(predict_labels(clf, X_train, y_train))
    print "F1 score for test set: {:.4f}.".format(predict_labels(clf, X_test, y_test))

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import MultinomialNB
classifiers = [RandomForestClassifier(), AdaBoostClassifier(), MultinomialNB()]
results = {
    'Classifier': [],
    'Size': [],
    'Train time': [],
    'predict time': [],
    'F1 score - train': [],
    'F1 score - test': []
}

datasets = [train_test_split(X_all, y_all, train_size=x, test_size=95) for x in [100, 200, 300]]

for clf in classifiers:
    for data in datasets:
        X_train, X_test, y_train, y_test = data
        time_train = train_classifier(clf, X_train, y_train)
        f1_train, time_predict = predict_labels(clf, X_train, y_train)
        f1_test, time_predict = predict_labels(clf, X_test, y_test)

        results['Classifier'].append(clf.__class__.__name__)
        results['Size'].append(X_train.shape[0])
        results['Train time'].append("{:.5f}".format(time_train))
        results['predict time'].append("{:.5f}".format(time_predict))
        results['F1 score - train'].append(f1_train)
        results['F1 score - test'].append(f1_test)
print pd.DataFrame(results)

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

# TODO: Execute the 'train_predict' function for each classifier and each training set size
# train_predict(clf, X_train, y_train, X_test, y_test)  
classifiers = [LogisticRegression(), SVC(), GaussianNB()]
datasets = [train_test_split(X_all, y_all, train_size=x, test_size=95) for x in [100, 200, 300]]

writer = pytablewriter.MarkdownTableWriter()

for clf in classifiers:
    results = {
        'Classifier': [],
        'Size': [],
        'Train time': [],
        'predict time': [],
        'F1 score - train': [],
        'F1 score - test': []
    }
    for data in datasets:
        X_train, X_test, y_train, y_test = data
        time_train = train_classifier(clf, X_train, y_train)
        f1_train, time_predict = predict_labels(clf, X_train, y_train)
        f1_test, time_predict = predict_labels(clf, X_test, y_test)
        results['Classifier'].append(clf.__class__.__name__)
        results['Size'].append(X_train.shape[0])
        results['Train time'].append("{:.5f}".format(time_train))
        results['predict time'].append("{:.5f}".format(time_predict))
        results['F1 score - train'].append(f1_train)
        results['F1 score - test'].append(f1_test)
    writer.from_dataframe(pd.DataFrame(results))
    writer.write_table()

# ### Tabular Results
# Edit the cell below to see how a table can be designed in [Markdown](https://github.com/adam-p/markdown-here/wiki/Markdown-Cheatsheet#tables). You can record your results from above in the tables provided.

# ** Classifer 1 - ?**  
# 
# | Training Set Size | Training Time | Prediction Time (test) | F1 Score (train) | F1 Score (test) |
# | :---------------: | :---------------------: | :--------------------: | :--------------: | :-------------: |
# | 100               |                         |                        |                  |                 |
# | 200               |        EXAMPLE          |                        |                  |                 |
# | 300               |                         |                        |                  |    EXAMPLE      |
# 
# ** Classifer 2 - ?**  
# 
# | Training Set Size | Training Time | Prediction Time (test) | F1 Score (train) | F1 Score (test) |
# | :---------------: | :---------------------: | :--------------------: | :--------------: | :-------------: |
# | 100               |                         |                        |                  |                 |
# | 200               |     EXAMPLE             |                        |                  |                 |
# | 300               |                         |                        |                  |     EXAMPLE     |
# 
# ** Classifer 3 - ?**  
# 
# | Training Set Size | Training Time | Prediction Time (test) | F1 Score (train) | F1 Score (test) |
# | :---------------: | :---------------------: | :--------------------: | :--------------: | :-------------: |
# | 100               |                         |                        |                  |                 |
# | 200               |                         |                        |                  |                 |
# | 300               |                         |                        |                  |                 |

# ## Choosing the Best Model
# In this final section, you will choose from the three supervised learning models the *best* model to use on the student data. You will then perform a grid search optimization for the model over the entire training set (`X_train` and `y_train`) by tuning at least one parameter to improve upon the untuned model's F<sub>1</sub> score. 

# ### Question 3 - Choosing the Best Model
# *Based on the experiments you performed earlier, in one to two paragraphs, explain to the board of supervisors what single model you chose as the best model. Which model is generally the most appropriate based on the available data, limited resources, cost, and performance?*

# **Answer: **

# ### Question 4 - Model in Layman's Terms
# *In one to two paragraphs, explain to the board of directors in layman's terms how the final model chosen is supposed to work. Be sure that you are describing the major qualities of the model, such as how the model is trained and how the model makes a prediction. Avoid using advanced mathematical or technical jargon, such as describing equations or discussing the algorithm implementation.*

# **Answer: **

# ### Implementation: Model Tuning
# Fine tune the chosen model. Use grid search (`GridSearchCV`) with at least one important parameter tuned with at least 3 different values. You will need to use the entire training set for this. In the code cell below, you will need to implement the following:
# - Import [`sklearn.grid_search.gridSearchCV`](http://scikit-learn.org/stable/modules/generated/sklearn.grid_search.GridSearchCV.html) and [`sklearn.metrics.make_scorer`](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.make_scorer.html).
# - Create a dictionary of parameters you wish to tune for the chosen model.
#  - Example: `parameters = {'parameter' : [list of values]}`.
# - Initialize the classifier you've chosen and store it in `clf`.
# - Create the F<sub>1</sub> scoring function using `make_scorer` and store it in `f1_scorer`.
#  - Set the `pos_label` parameter to the correct value!
# - Perform grid search on the classifier `clf` using `f1_scorer` as the scoring method, and store it in `grid_obj`.
# - Fit the grid search object to the training data (`X_train`, `y_train`), and store it in `grid_obj`.

# In[ ]:


# TODO: Import 'GridSearchCV' and 'make_scorer'

# TODO: Create the parameters list you wish to tune
parameters = None

# TODO: Initialize the classifier
clf = None

# TODO: Make an f1 scoring function using 'make_scorer' 
f1_scorer = None

# TODO: Perform grid search on the classifier using the f1_scorer as the scoring method
grid_obj = None

# TODO: Fit the grid search object to the training data and find the optimal parameters
grid_obj = None

# Get the estimator
#clf = grid_obj.best_estimator_

# Report the final F1 score for training and testing after parameter tuning
#print "Tuned model has a training F1 score of {:.4f}.".format(predict_labels(clf, X_train, y_train))
#print "Tuned model has a testing F1 score of {:.4f}.".format(predict_labels(clf, X_test, y_test))

# ### Question 5 - Final F<sub>1</sub> Score
# *What is the final model's F<sub>1</sub> score for training and testing? How does that score compare to the untuned model?*

# **Answer: **

# > **Note**: Once you have completed all of the code implementations and successfully answered each question above, you may finalize your work by exporting the iPython Notebook as an HTML document. You can do this by using the menu above and navigating to  
