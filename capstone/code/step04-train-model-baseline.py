#!/usr/bin/env python

from sklearn import linear_model

from regressionutil import *

def main():
    X_train_origin, X_test_origin, y_train_origin, y_test_origin = get_data_tuple()
    sizeTrain = 51846 # 51846
    sizeTest = 22221 # 22221
    X_train, X_test, y_train, y_test = \
        X_train_origin[:sizeTrain], X_test_origin[:sizeTest], y_train_origin[:sizeTrain], y_test_origin[:sizeTest]
    clf = linear_model.LinearRegression()
    time_train = train_classifier(clf, X_train, y_train)
    print('Linear Regression training, dur=%s' % time_train)
    train_pred, train_dur = predict_labels(clf, X_train)
    train_score = calculate_metrics(y_train, train_pred)
    test_pred, test_dur = predict_labels(clf, X_test)
    test_score = calculate_metrics(y_test, test_pred)
    print('trainSet score=%.6f, dur=%s' % (train_score, train_dur))
    print('testSet score=%.6f, dur=%s' % (test_score, test_dur))
    """
    Linear Regression training, dur=0.0639629364014
    trainSet score=0.488282, dur=0.00290894508362
    testSet  score=0.488518, dur=0.000833034515381
    """
    submit_result_test('04.linearreg', clf)

#
# 0.43192 #1
# 2018-11-01 0.49332 No.1459
#
#
if __name__ == '__main__':
    main()
