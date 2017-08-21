#!/usr/bin/env python

import numpy as np
import pandas as pd
import time
from sklearn.metrics import f1_score

def train_classifier(clf, X_train, y_train):
    print "Training {}...".format(clf.__class__.__name__)
    start = time.time()
    clf.fit(X_train, y_train)
    end = time.time()
    print "Done!\nTraining time (secs): {:.6f}".format(end - start)
    return end - start

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


def predict_labels(clf, features, target):
    ''' Makes predictions using a fit classifier based on F1 score. '''

    # Start the clock, make predictions, then stop the clock
    start = time.time()
    y_pred = clf.predict(features)
    end = time.time()

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


# Read student data
student_data = pd.read_csv("student-data.csv")
print(student_data.shape)

# Total number of students: 395
# Number of features: 30
# Number of students who passed: 265
# Number of students who failed: 130
# Graduation rate of the class: 67.09%

# Extract target column 'passed'
target_col = student_data.columns[-1]
y_all = student_data[target_col]

# Separate the data into feature data and target data (X_all and y_all, respectively)
feature_cols = list(student_data.columns[:-1])
X_all = student_data[feature_cols]
X_all = preprocess_features(X_all)
print "Processed feature columns ({} total features):\n{}".format(len(X_all.columns), list(X_all.columns))
print X_all.head()


# Split training and test
from sklearn.model_selection import train_test_split

# TODO: Set the number of training points
num_train = 300 # about 75% of the data

# Set the number of testing points
num_test = X_all.shape[0] - num_train

# TODO: Shuffle and split the dataset into the number of training and testing points above
X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=num_test)
# Show the results of the split
print "Training set has {} samples.".format(X_train.shape[0])
print "Testing set has {} samples.".format(X_test.shape[0])


# TODO: Import the three supervised learning models from sklearn
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

# TODO: Initialize the three models
clf_A = DecisionTreeClassifier()
clf_B = SVC()
clf_C = GaussianNB()

# TODO: Set up the training set sizes

# TODO: Execute the 'train_predict' function for each classifier and each training set size
# train_predict(clf, X_train, y_train, X_test, y_test)
#for l in (100,200,300):
#    train_predict(clf_A,X_train[:l], y_train[:l], X_test, y_test)
#    train_predict(clf_B,X_train[:l], y_train[:l], X_test, y_test)
#    train_predict(clf_C,X_train[:l], y_train[:l], X_test, y_test)

classifiers = [DecisionTreeClassifier(), SVC(), GaussianNB()]
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
        results['Train time'].append("{:.3f}".format(time_train))
        results['predict time'].append("{:.3f}".format(time_predict))
        results['F1 score - train'].append(f1_train)
        results['F1 score - test'].append(f1_test)

print pd.DataFrame(results)