#!/usr/bin/env python

import time

import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error

from regressionutil import submit_result_test, get_data

from sklearn import svm
from sklearn import tree
from sklearn import ensemble

def trainModel(data, name, regressor):
    time_start = time.time()
    # {'train': {'X': X_train, 'y': y_train}, 'test': {'X': X_test, 'y': y_test}}
    regressor.fit(data['train']['X'], data['train']['y'])
    y_pred = regressor.predict(data['test']['X'])
    rmse = np.sqrt(mean_squared_error(y_pred, data['test']['y']))
    time_end = time.time()
    return (rmse, time_end - time_start)

def main():
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


#
# 0.43192 #1
# 2018-11-01 0.49332 No.1459
#
#
if __name__ == '__main__':
    main()