#!/usr/bin/env python

import os
import time

import numpy as np
import pandas as pd

from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot as plt

RANDOM_STATE = 42
NUM_TRAIN_DATA = 74067

WORK_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
DATA_DIR = os.path.realpath(WORK_DIR + '/features')

def linearRegression():
    data_all = pd.read_csv(DATA_DIR + '/data.csv.gz', encoding="ISO-8859-1")
    data = data_all.iloc[:NUM_TRAIN_DATA]
    X = data.drop(["relevance", "id"], axis=1)
    y = data["relevance"].copy()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=RANDOM_STATE)
    print X_train.shape, y_train.shape
    print X_test.shape, y_test.shape
    regressor = linear_model.LinearRegression()
    regressor.fit(X_train, y_train)
    y_pred = regressor.predict(X_test)
    print('Liner Regression R squared: %.4f' % regressor.score(X_test, y_test))
    plt.scatter(y_test, y_pred)
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    lin_mse = mean_squared_error(y_pred, y_test)
    lin_rmse = np.sqrt(lin_mse)
    print('Liner Regression RMSE: %.4f' % lin_rmse)
    plt.savefig(DATA_DIR + '/linearregress.png')
    plt.show()
    """
    Liner Regression R squared: 0.1453
    Liner Regression RMSE: 0.4927
    """
    return regressor

def submit(regressor):
    data_all = pd.read_csv(DATA_DIR + '/data.csv.gz', encoding="ISO-8859-1")
    data_submit = data_all.iloc[NUM_TRAIN_DATA:]
    submit_id = data_submit['id']
    submit_test = data_submit.drop(["relevance", "id"], axis=1)
    submit_pred = regressor.predict(submit_test)
    pd.DataFrame({"id": submit_id, "relevance": submit_pred})\
        .to_csv(DATA_DIR + '/submission.csv.gz', index=False, compression='gzip')

def trainModel():
    time_start = time.time()
    regressor = linearRegression()
    submit(regressor)
    time_end = time.time()
    print("--- Took: %s minutes ---" % round(((time_end - time_start) / 60), 2))


#
# 0.43192 #1
# 2018-11-01 0.49332 No.1459
#
#
if __name__ == '__main__':
    trainModel()
