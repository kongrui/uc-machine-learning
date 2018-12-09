#!/usr/bin/env python

import os
import time

import numpy as np
import pandas as pd

#from sklearn.ensemble import RandomForestRegressor, BaggingRegressor
#from sklearn.metrics import mean_squared_error
#from sklearn.model_selection import train_test_split

#from sklearn import pipeline,model_selection
#from sklearn.metrics import mean_squared_error, make_scorer

RANDOM_STATE = 42
NUM_TRAIN_DATA = 74067

WORK_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
DATA_DIR = os.path.realpath(WORK_DIR + '/features')

def fmean_squared_error(ground_truth, predictions):
    fmean_squared_error_ = mean_squared_error(ground_truth, predictions)**0.5
    return fmean_squared_error_

RSME  = make_scorer(fmean_squared_error, greater_is_better=False)

def trainModel():
    time_start = time.time()

    df_data = pd.read_csv(DATA_DIR + '/data.csv.gz', encoding="ISO-8859-1")

    y = df_data['relevance'].values
    X = df_data.drop(['relevance'], axis=1).values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=RANDOM_STATE)

    rf = RandomForestRegressor(n_estimators=15, max_depth=6, random_state=0)
    #clf = BaggingRegressor(rf, n_estimators=45, max_samples=0.1, random_state=RANDOM_STATE)
    rf.fit(X_train, y_train)

    time_fit = time.time()
    print("Fit... dur=%s" % round(((time_fit - time_start) / 60), 2))

    y_pred = rf.predict(X_test)
    # Compute and print R^2 and RMSE
    print("R^2: {}".format(rf.score(X_test, y_test)))
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print("Root Mean Squared Error: {}".format(rmse))

    plt.scatter(y_test, y_pred)
    plt.xlabel('True Values')
    plt.ylabel('Predictions')


if __name__ == '__main__':
    trainModel()
