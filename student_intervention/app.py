#!/usr/bin/env python

import numpy as np
import pandas as pd
from time import time
from sklearn.metrics import f1_score


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
