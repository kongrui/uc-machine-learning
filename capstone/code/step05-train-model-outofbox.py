#!/usr/bin/env python

from sklearn import ensemble
from sklearn import linear_model
from sklearn import tree

from regressionutil import *
from regressionutil import NUM_TRAIN_DATA
from regressionutil import RANDOM_STATE

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
        , ensemble.RandomForestRegressor(random_state=RANDOM_STATE)
    ]

    X_train_origin, X_test_origin, y_train_origin, y_test_origin = get_data_tuple()

    for clf in classifiers:
        results = {
            'Classifier': [],
            'size': [],
            'time - train': [],
            'time - predict': [],
            'score - train': [],
            'score - test': []
        }
        sizeTest = 22221  # 22221 sizeTrain = 51846  # 51846
        for size in [5000, 10000, NUM_TRAIN_DATA]:
            X_train, X_test, y_train, y_test = \
                X_train_origin[:size], X_test_origin[:sizeTest] \
                    , y_train_origin[:size], y_test_origin[:sizeTest]
            time_train = train_classifier(clf, X_train, y_train)
            train_pred, train_dur = predict_labels(clf, X_train)
            train_score = calculate_metrics(y_train, train_pred)
            test_pred, test_dur = predict_labels(clf, X_test)
            test_score = calculate_metrics(y_test, test_pred)
            results['Classifier'].append(clf.__class__.__name__)
            results['size'].append(size)
            results['time - train'].append("{:.4f}".format(train_dur))
            results['time - predict'].append("{:.4f}".format(test_dur))
            results['score - train'].append(train_score)
            results['score - test'].append(test_score)
        df = pd.DataFrame(results)
        print(df.to_csv(index=False, sep='|'))


"""
Classifier|score - test|score - train|size|time - predict|time - train
LinearRegression|0.4888664177838271|0.4949631820999136|5000|0.0014|0.0030
LinearRegression|0.4888296261069622|0.4921368308721272|10000|0.0017|0.0015
LinearRegression|0.48851831932878437|0.48828185422433157|74067|0.0006|0.0040

Classifier|score - test|score - train|size|time - predict|time - train
DecisionTreeRegressor|0.694166418491208|0.011547294055318761|5000|0.0047|0.0016
DecisionTreeRegressor|0.6822399200039269|0.03208177052470764|10000|0.0048|0.0025
DecisionTreeRegressor|0.6840096861311904|0.06677815630290426|74067|0.0070|0.0190

Classifier|score - test|score - train|size|time - predict|time - train
GradientBoostingRegressor|0.48438658687074404|0.4674867633928048|5000|0.0356|0.0100
GradientBoostingRegressor|0.4827951980395866|0.472821443715223|10000|0.0249|0.0111
GradientBoostingRegressor|0.48007840339553226|0.47806534768029424|74067|0.0315|0.0918

Classifier|score - test|score - train|size|time - predict|time - train
RandomForestRegressor|0.5243974138545319|0.2244114168698989|5000|0.0314|0.0106
RandomForestRegressor|0.5230794816950426|0.2238573140167452|10000|0.0370|0.0181
RandomForestRegressor|0.5216206509872446|0.22546386042857158|74067|0.0558|0.1391
"""

if __name__ == '__main__':
    tryFewModels()
