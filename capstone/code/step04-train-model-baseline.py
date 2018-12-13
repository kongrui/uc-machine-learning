#!/usr/bin/env python

import time

import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error

from regressionutil import submit_result_test, get_data

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
    regressor = linear_model.LinearRegression()
    rmse, dur = trainModel(data, 'linear', regressor)
    print('Linear Regression RMSE=%.6f, dur=%s' % (rmse, dur))
    """
    Linear Regression RMSE: 0.4927
    """
    submit_result_test('04.linearreg', regressor)

#
# 0.43192 #1
# 2018-11-01 0.49332 No.1459
#
#
if __name__ == '__main__':
    main()
