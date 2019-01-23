#!/usr/bin/env python


import pandas as pd

from sklearn import ensemble

from sklearn import model_selection
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score

from regressionutil import *

from scipy import stats

"""
(51846, 10) (51846,)
(22221, 10) (22221,)
Regression training, dur=2.56662201881
trainSet score=0.478065, dur=0.0721879005432
testSet score=0.480078, dur=0.0311019420624
Feature Importances
[0.23348383 0.10495558 0.11530168 0.11945179 0.01654673 0.02605085
 0.05854739 0.0385929  0.20169511 0.08537413]
"""
def evalulateModel(clf):
    X_train_origin, X_test_origin, y_train_origin, y_test_origin = get_data_tuple()
    sizeTrain = 51846 # 51846
    sizeTest = 22221 # 22221
    X_train, X_test, y_train, y_test = \
        X_train_origin[:sizeTrain], X_test_origin[:sizeTest], y_train_origin[:sizeTrain], y_test_origin[:sizeTest]
    time_train = train_classifier(clf, X_train, y_train)
    print('Regression training, dur=%s' % time_train)
    train_pred, train_dur = predict_labels(clf, X_train)
    train_score = calculate_metrics(y_train, train_pred)
    test_pred, test_dur = predict_labels(clf, X_test)
    test_score = calculate_metrics(y_test, test_pred)
    print('trainSet score=%.6f, dur=%s' % (train_score, train_dur))
    print('testSet score=%.6f, dur=%s' % (test_score, test_dur))
    feature_importances = pd.DataFrame(clf.feature_importances_,
                                       index=X_train.columns,
                                       columns=['importance']).sort_values('importance', ascending=False)
    print feature_importances


def evaluateModelRobust(clf):

    rmse_scorer = make_scorer(calculate_metrics, greater_is_better=False)
    data_all = pd.read_csv(DATA_DIR + '/data.csv.gz', encoding="ISO-8859-1")
    data = data_all.iloc[:NUM_TRAIN_DATA]
    X = data.drop(["relevance", "id"], axis=1)
    y = data["relevance"].copy()
    score = cross_val_score(clf, X, y, cv=11, scoring=rmse_scorer)
    print(score)
    print(stats.describe(score))

    X_train_origin, X_test_origin, y_train_origin, y_test_origin = get_data_tuple()

    sizeTest = 22221  # 22221 sizeTrain = 51846  # 51846
    results = {
        'size': [],
        'time - train': [],
        'time - predict': [],
        'score - train': [],
        'score - test': []
    }
    for size in [ 1000, 5000, 10000, 20000, 30000, 40000, 50000, 60000, NUM_TRAIN_DATA ]:
       X_train, X_test, y_train, y_test = \
           X_train_origin[:size], X_test_origin[:sizeTest] \
               , y_train_origin[:size], y_test_origin[:sizeTest]
       time_train = train_classifier(clf, X_train, y_train)
       train_pred, train_dur = predict_labels(clf, X_train)
       train_score = calculate_metrics(y_train, train_pred)
       test_pred, test_dur = predict_labels(clf, X_test)
       test_score = calculate_metrics(y_test, test_pred)
       results['size'].append(size)
       results['time - train'].append("{:.4f}".format(train_dur))
       results['time - predict'].append("{:.4f}".format(test_dur))
       results['score - train'].append(train_score)
       results['score - test'].append(test_score)

    df = pd.DataFrame(results)
    print(df.to_csv(index=False, sep='|'))




def tuningParameters(param_grid):

    scorer = make_scorer(calculate_metrics, greater_is_better=False)

    X_train_origin, X_test_origin, y_train_origin, y_test_origin = get_data_tuple()
    sizeTrain = 51846 # 51846
    sizeTest = 22221 # 22221
    X_train, X_test, y_train, y_test = \
        X_train_origin[:sizeTrain], X_test_origin[:sizeTest], y_train_origin[:sizeTrain], y_test_origin[:sizeTest]

    gbr = ensemble.GradientBoostingRegressor(random_state=RANDOM_STATE)
    clf = model_selection.GridSearchCV(estimator=gbr, param_grid=param_grid, n_jobs=1, cv=10, scoring=scorer)
    time_train = train_classifier(clf, X_train, y_train)
    print('Regression training, dur=%s' % time_train)
    train_pred, train_dur = predict_labels(clf, X_train)
    train_score = calculate_metrics(y_train, train_pred)
    test_pred, test_dur = predict_labels(clf, X_test)
    test_score = calculate_metrics(y_test, test_pred)
    print('trainSet score=%.6f, dur=%s' % (train_score, train_dur))
    print('testSet score=%.6f, dur=%s' % (test_score, test_dur))
    print "parameters"
    print(clf.best_params_)
    print('Best CV Score:')
    print(clf.best_score_)
    return clf

def attempt1():
    param_grid = {
        'n_estimators': [100]
    }
    tuningParameters(param_grid)

def attempt2():
    param_grid = {
        'n_estimators': [15, 45, 70],
        'max_features': [4, 6, 8],
        'max_depth': [6, 8],
        'learning_rate': [0.1],
        'min_samples_leaf': [50],
        'subsample': [0.8]
    }

    tuningParameters(param_grid)
"""

(51846, 10) (51846,)
(22221, 10) (22221,)
Regression training, dur=318.872573137
trainSet score=0.473212, dur=0.0793888568878
testSet score=0.479474, dur=0.0314331054688
Feature Importances
{'learning_rate': 0.1, 'min_samples_leaf': 50, 'n_estimators': 45, 'subsample': 0.8, 'max_features': 6, 'max_depth': 6}
Best CV Score:
-0.4799740177071018
"""

def attempt3():
    param_grid = {
        'n_estimators': [40, 45, 50],
        'max_features': [4, 6, 8, 10],
        'max_depth': [6, 8],
        'learning_rate': [0.1],
        'min_samples_leaf': [50],
        'subsample': [0.8]
    }

    tuningParameters(param_grid)
"""
(51846, 10) (51846,)
(22221, 10) (22221,)
Regression training, dur=715.543130875
trainSet score=0.473896, dur=0.0749540328979
testSet score=0.479311, dur=0.0319809913635
Feature Importances
{'learning_rate': 0.1, 'min_samples_leaf': 50, 'n_estimators': 45, 'subsample': 0.8, 'max_features': 4, 'max_depth': 6}
Best CV Score:
-0.4798894154650474
"""

def attempt4():
    param_grid = {
        'n_estimators': [40, 45, 50],
        'max_features': [3, 4, 5, 6],
        'max_depth': [4, 5, 6, 7, 8],
        'learning_rate': [0.1],
        'min_samples_leaf': [50],
        'subsample': [0.8]
    }

    tuningParameters(param_grid)
"""
(51846, 10) (51846,)
(22221, 10) (22221,)
Regression training, dur=660.013599873
trainSet score=0.473896, dur=0.0668947696686
testSet score=0.479311, dur=0.0287511348724
Feature Importances
{'learning_rate': 0.1, 'min_samples_leaf': 50, 'n_estimators': 45, 'subsample': 0.8, 'max_features': 4, 'max_depth': 6}
Best CV Score:
-0.4798894154650474
"""

def attempt5():
  param_grid = {
        'n_estimators': [25, 45, 55],
        'max_features': [6, 7, 8],
        'max_depth': [2, 3, 4],
        'learning_rate': [0.1],
        'min_samples_leaf': [50, 75, 100],
        'subsample': [0.8]
  }
  tuningParameters(param_grid)

def beforeTuningParameter():
    clf = ensemble.GradientBoostingRegressor(random_state=RANDOM_STATE, n_estimators=100)
    evalulateModel(clf)
    return clf

def afterTuningParameter():
    clf = ensemble.GradientBoostingRegressor(random_state=RANDOM_STATE,
                                             learning_rate=0.1, min_samples_leaf=50, n_estimators=45, subsample=0.8,
                                             max_features=4, max_depth=6)
    evalulateModel(clf)
    evaluateModelRobust(clf)
    return clf

def tuningOnsingleParameter(x_train, x_test, y_train, y_test, paramName, choices):
  train_results = []
  test_results = []
  for eta in choices:
    params = {}
    params[paramName] = eta
    model = ensemble.GradientBoostingRegressor(**params)
    train_classifier(model, x_train, y_train)
    train_pred, dur = predict_labels(model, x_train)
    test_pred, dur = predict_labels(model, x_test)
    train_results.append(calculate_metrics(y_train, train_pred))
    test_results.append(calculate_metrics(y_test, test_pred))
  l1, = plt.plot(choices, train_results, 'b', label="Train")
  l2, = plt.plot(choices, test_results, 'r', label="Test")
  plt.ylabel('rmse')
  plt.xlabel(paramName)
  plt.legend((l1, l2), ('Train', 'Test'), loc='best')
  plt.savefig(IMG_DIR + '/07.tuning-gb.' + paramName + '.png')
  # plt.show()

def tuneLearnRate(X_train, X_test, y_train, y_test):
  choices = [0.01, 0.02, 0.03, 0.04, 0.05, 0.6]
  paramName = 'learning_rate'
  tuningOnsingleParameter(X_train, X_test, y_train, y_test, paramName, choices)

def tuneNEstimators(X_train, X_test, y_train, y_test):
  paramName = 'n_estimators'
  choices = [1, 2, 4, 8, 16, 32, 64, 100, 200]
  tuningOnsingleParameter(X_train, X_test, y_train, y_test, paramName, choices)

def tuneMaxDepth(X_train, X_test, y_train, y_test):
  paramName = 'max_depth'
  choices = [ 1., 2., 3., 4.,  5.,  6.,  7.,  8.,  9., 10.]
  tuningOnsingleParameter(X_train, X_test, y_train, y_test, paramName, choices)

def tuneMinSampleSplit(X_train, X_test, y_train, y_test):
  paramName = 'min_samples_split'
  choices = [5, 10, 20 , 30, 40, 50, 100]
  tuningOnsingleParameter(X_train, X_test, y_train, y_test, paramName, choices)

def tuneMinSampleLeaf(X_train, X_test, y_train, y_test):
  paramName = 'min_samples_leaf'
  choices = [1, 20, 50, 100, 200, 500, 1000, 2000]
  tuningOnsingleParameter(X_train, X_test, y_train, y_test, paramName, choices)

def tuneMaxFeatures(X_train, X_test, y_train, y_test):
  paramName = 'max_features'
  choices = list(range(1, X_train.shape[1]))
  tuningOnsingleParameter(X_train, X_test, y_train, y_test, paramName, choices)

def tuneSubSamples(X_train, X_test, y_train, y_test):
  paramName = 'subsample'
  choices = [0.05, 0.1, 0.2, 0.5, 0.8, 0.9]
  tuningOnsingleParameter(X_train, X_test, y_train, y_test, paramName, choices)

if __name__ == '__main__':
  X_train, X_test, y_train, y_test = get_data_tuple()

  #
  #tuneLearnRate(X_train, X_test, y_train, y_test)
  #tuneNEstimators(X_train, X_test, y_train, y_test)
  #tuneMaxDepth(X_train, X_test, y_train, y_test)
  #tuneMinSampleSplit(X_train, X_test, y_train, y_test)
  #tuneMinSampleLeaf(X_train, X_test, y_train, y_test)
  #tuneMaxFeatures(X_train, X_test, y_train, y_test)
  #tuneSubSamples(X_train, X_test, y_train, y_test)

  #attempt2()
  #attempt3()
  #attempt4()
  afterTuningParameter()
  #submit_result_test('gb.aft', clf)
  """
Regression training, dur=1.85773301125
trainSet score=0.473896, dur=0.0938727855682
testSet score=0.479311, dur=0.0370998382568
                  importance
product_uid         0.214745
term_ratio_title    0.198089
len_desc            0.141595
term_ratio_desc     0.135161
len_title           0.104772
term_feq_title      0.079104
len_query           0.073408
term_feq_desc       0.026005
query_feq_title     0.013829
query_feq_desc      0.013292
   """