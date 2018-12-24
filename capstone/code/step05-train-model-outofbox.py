#!/usr/bin/env python

import time

import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error

from regressionutil import submit_result_test, get_data, get_data_tuple

from sklearn import svm
from sklearn import tree
from sklearn import ensemble
from sklearn import naive_bayes

from regressionutil import DATA_DIR
from regressionutil import IMG_DIR
from regressionutil import NUM_TRAIN_DATA
from regressionutil import RANDOM_STATE

#def trainModel(data, name, regressor):
#    time_start = time.time()
#    regressor.fit(data['train']['X'], data['train']['y'])
#    y_pred = regressor.predict(data['test']['X'])
#    rmse = np.sqrt(mean_squared_error(y_pred, data['test']['y']))
#    time_end = time.time()
#    return (rmse, time_end - time_start)

def train_classifier(clf, X_train, y_train):
    start = time.time()
    clf.fit(X_train, y_train)
    end = time.time()
    return end - start

def predict_labels(clf, features, target):
    start = time.time()
    y_pred = clf.predict(features)
    end = time.time()
    print "Made predictions in {:.4f} seconds.".format(end - start)
    score = np.sqrt(mean_squared_error(target, y_pred))
    dur = end - start
    return score, dur

def tryFewAlgorithms():
    data = get_data()

    #
    #regressor = svm.SVR(kernel='rbf', gamma='auto')
    #rmse, dur = trainModel(data, 'svm', regressor)
    #print('SVM Regression RMSE=%.6f, dur=%s' % (rmse, dur))
    """
    SVM Regression RMSE=0.544827, dur=228.630357981
    low and 
    """

    #
    #regressor = tree.DecisionTreeRegressor(random_state = 0)
    #rmse, dur = trainModel(data, 'dt', regressor)
    #print('DT Regression RMSE=%.6f, dur=%s' % (rmse, dur))
    """
    DT Regression RMSE=0.683829, dur=0.518572092056
    """

    #regressor = ensemble.RandomForestRegressor(random_state = 0)
    #rmse, dur = trainModel(data, 'rf', regressor)
    #print('RF Regression RMSE=%.6f, dur=%s' % (rmse, dur))
    """
    RF Regression RMSE=0.519509, dur=2.308355093
    """

    regressor = ensemble.GradientBoostingRegressor(random_state = 0)
    rmse, dur = trainModel(data, 'rf', regressor)
    print('GB Regression RMSE=%.6f, dur=%s' % (rmse, dur))
    """
    GB Regression RMSE=0.480073, dur=2.14031505585
    """

def train_classifier(clf, X_train, y_train):
    start = time.time()
    clf.fit(X_train, y_train)
    end = time.time()
    return end - start

def tryFewModels():

    classifiers = [
        linear_model.LinearRegression()
        , tree.DecisionTreeRegressor(random_state=RANDOM_STATE)
        , ensemble.GradientBoostingRegressor(random_state=RANDOM_STATE)
        , ensemble.RandomForestRegressor(random_state = RANDOM_STATE)
    ]

    X_train, X_test, y_train, y_test = get_data_tuple()

    for clf in classifiers:
        results = {
            'Classifier': [],
            'size': [],
            'time - train': [],
            'time - predict': [],
            'score - train': [],
            'score - test': []
        }
        #for size in [1000, 5000, 10000, 20000, 30000, 40000, 50000, 60000]:
        for size in [1000, 5000, 10000]:
            time_train = train_classifier(clf, X_train[:size], y_train[:size])
            score_train, time_predict = predict_labels(clf, X_train[:size], y_train[:size])
            score_test, time_predict = predict_labels(clf, X_test, y_test)
            results['Classifier'].append(clf.__class__.__name__)
            results['size'].append(size)
            results['time - train'].append("{:.4f}".format(time_train))
            results['time - predict'].append("{:.4f}".format(time_predict))
            results['score - train'].append(score_train)
            results['score - test'].append(score_test)
        df = pd.DataFrame(results)
        print(df.to_csv(index=False, sep='|'))


#
# 0.43192 #1
# 2018-11-01 0.49332 No.1459
#
#
if __name__ == '__main__':
    tryFewModels()