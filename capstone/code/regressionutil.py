#!/usr/bin/env python

import time
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('agg')
matplotlib.use('Qt5Agg')

from matplotlib import pyplot as plt

RANDOM_STATE = 42
NUM_TRAIN_DATA = 74067
PCT_TEST_DATA_SIZE = 0.3

WORK_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
RAW_DIR = os.path.realpath(WORK_DIR + '/data')
DATA_DIR = os.path.realpath(WORK_DIR + '/features')
IMG_DIR = os.path.realpath(WORK_DIR + '/images')
TEMP_DIR = os.path.realpath(WORK_DIR + '/temp')
if not os.path.exists(TEMP_DIR):
    os.makedirs(TEMP_DIR)

def performance_metric(y_predict, y_test):
    mse = mean_squared_error(y_predict, y_test)
    return np.sqrt(mse)

def get_data_tuple():
    data_all = pd.read_csv(DATA_DIR + '/data.csv.gz', encoding="ISO-8859-1")
    data = data_all.iloc[:NUM_TRAIN_DATA]
    X = data.drop(["relevance", "id"], axis=1)
    y = data["relevance"].copy()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=PCT_TEST_DATA_SIZE, random_state=RANDOM_STATE)
    print X_train.shape, y_train.shape
    print X_test.shape, y_test.shape
    return X_train, X_test, y_train, y_test

def get_data():
    X_train, X_test, y_train, y_test = get_data_tuple()
    data = {'train': {'X': X_train, 'y': y_train}, 'test': {'X': X_test, 'y': y_test}}
    #scaler = StandardScaler().fit(data['train']['X'])
    #data['train']['X'] = scaler.transform(data['train']['X'])
    #data['test']['X'] = scaler.transform(data['test']['X'])
    return data

def plot_result(name, y_test, y_pred):
    plt.scatter(y_test, y_pred)
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.savefig(IMG_DIR + '/' + name + '.png')
    plt.show()

def train_classifier(clf, X_train, y_train):
    start = time.time()
    clf.fit(X_train, y_train)
    end = time.time()
    return end - start

def predict_labels(clf, features):
    start = time.time()
    y_pred = clf.predict(features)
    end = time.time()
    dur = end - start
    return y_pred, dur

def calculate_metrics(y_true, y_pred):
    score = np.sqrt(mean_squared_error(y_true, y_pred))
    return score

def submit_result_test(name, regressor):
    data_all = pd.read_csv(DATA_DIR + '/data.csv.gz', encoding="ISO-8859-1")
    data_submit = data_all.iloc[NUM_TRAIN_DATA:]
    submit_id = data_submit['id']
    submit_test = data_submit.drop(["relevance", "id"], axis=1)
    submit_pred = regressor.predict(submit_test)
    pd.DataFrame({"id": submit_id, "relevance": submit_pred}) \
       .to_csv(TEMP_DIR + '/' + name + '.submission.csv.gz', index=False, compression='gzip')
