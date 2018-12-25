#!/usr/bin/env python

from sklearn import ensemble

from sklearn import model_selection
from sklearn.metrics import make_scorer

from regressionutil import *

def firstAttempt():
    X_train_origin, X_test_origin, y_train_origin, y_test_origin = get_data_tuple()
    sizeTrain = 51846 # 51846
    sizeTest = 22221 # 22221
    X_train, X_test, y_train, y_test = \
        X_train_origin[:sizeTrain], X_test_origin[:sizeTest], y_train_origin[:sizeTrain], y_test_origin[:sizeTest]
    clf = ensemble.GradientBoostingRegressor(random_state=RANDOM_STATE, n_estimators=100)
    time_train = train_classifier(clf, X_train, y_train)
    print('Regression training, dur=%s' % time_train)
    train_pred, train_dur = predict_labels(clf, X_train)
    train_score = calculate_metrics(y_train, train_pred)
    test_pred, test_dur = predict_labels(clf, X_test)
    test_score = calculate_metrics(y_test, test_pred)
    print('trainSet score=%.6f, dur=%s' % (train_score, train_dur))
    print('testSet score=%.6f, dur=%s' % (test_score, test_dur))
    print "Feature Importances"
    print clf.feature_importances_

def secondAttempt():

    scorer = make_scorer(calculate_metrics, greater_is_better=False)

    X_train_origin, X_test_origin, y_train_origin, y_test_origin = get_data_tuple()
    sizeTrain = 51846 # 51846
    sizeTest = 22221 # 22221
    X_train, X_test, y_train, y_test = \
        X_train_origin[:sizeTrain], X_test_origin[:sizeTest], y_train_origin[:sizeTrain], y_test_origin[:sizeTest]

    gbr = ensemble.GradientBoostingRegressor(random_state=RANDOM_STATE)
    param_grid = {
        'n_estimators': [15, 45, 70],
        'max_features': [6, 8],
        'max_depth': [6, 8],
        'learning_rate': [0.1],
        'min_samples_leaf' : [50],
        'subsample': [0.8]
    }
    clf = model_selection.GridSearchCV(estimator=gbr, param_grid=param_grid, n_jobs=1, cv=10, scoring=scorer)
    time_train = train_classifier(clf, X_train, y_train)
    print('Regression training, dur=%s' % time_train)
    train_pred, train_dur = predict_labels(clf, X_train)
    train_score = calculate_metrics(y_train, train_pred)
    test_pred, test_dur = predict_labels(clf, X_test)
    test_score = calculate_metrics(y_test, test_pred)
    print('trainSet score=%.6f, dur=%s' % (train_score, train_dur))
    print('testSet score=%.6f, dur=%s' % (test_score, test_dur))
    print "Feature Importances"
    print(clf.best_params_)
    print('Best CV Score:')
    print(clf.best_score_)

#
if __name__ == '__main__':
    firstAttempt()
    secondAttempt()
