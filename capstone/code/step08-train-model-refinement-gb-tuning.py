from sklearn import ensemble

from sklearn import model_selection
from sklearn.metrics import make_scorer

from regressionutil import *

from sklearn import ensemble

from sklearn import model_selection
from sklearn.metrics import make_scorer

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_curve, auc

from matplotlib.legend_handler import HandlerLine2D

# Let's first fit a gradient boosting classifier with default parameters to get a baseline idea of the performance
# learning rate shrinks the contribution of each tree by learning_rate.
# n_estimators represents the number of trees in the forest. Usually the higher the number of trees the better to learn the data.
# However, adding a lot of trees can slow down the training process considerably, therefore we do a parameter search to find the sweet spot.
# max_depth. This indicates how deep the built tree can be. The deeper the tree, the more splits it has and it captures more information about how the data. We fit a decision tree with depths ranging from 1 to 32 and plot the training and test errors.

def tuneLearnRate(x_train, x_test, y_train, y_test):
  learning_rates = [0.01, 0.02, 0.03, 0.04, 0.05, 0.6]
  train_results = []
  test_results = []
  for eta in learning_rates:
    model = ensemble.GradientBoostingRegressor(learning_rate=eta)
    train_classifier(model, x_train, y_train)
    train_pred, dur = predict_labels(model, x_train)
    test_pred, dur = predict_labels(model, x_test)
    train_results.append(calculate_metrics(y_train, train_pred))
    test_results.append(calculate_metrics(y_test, test_pred))
  l1, = plt.plot(learning_rates, train_results, 'b', label="Train")
  l2, = plt.plot(learning_rates, test_results, 'r', label="Test")
  plt.ylabel('rmse')
  plt.xlabel('learning rate')
  plt.legend((l1, l2), ('Train', 'Test'), loc='best')
  plt.show()

def tuneNEstimators(x_train, x_test, y_train, y_test):
  n_estimators = [1, 2, 4, 8, 16, 32, 64, 100, 200]

def tuneMaxDepth(x_train, x_test, y_train, y_test):
  max_depth = np.linspace(1, 32, 32, endpoint=True)
  plt.xlabel('Tree depth')
  plt.show()

# min_samples_split
# min_samples_split represents the minimum number of samples required to split an internal node.
# This can vary between considering at least one sample at each node to considering all of the samples at each node.
# When we increase this parameter, the tree becomes more constrained as it has to consider more samples at each node.
# Here we will vary the parameter from 10% to 100% of the samples
def tuneMinSampleSplit(x_train, x_test, y_train, y_test):
  min_samples_splits = np.linspace(0.1, 1.0, 10, endpoint=True)
  train_results = []
  test_results = []
  for min_samples_split in min_samples_splits:
    model = GradientBoostingClassifier(min_samples_split=min_samples_split)
    model.fit(x_train, y_train)
    train_pred = model.predict(x_train)
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, train_pred)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    train_results.append(roc_auc)
    y_pred = model.predict(x_test)
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    test_results.append(roc_auc)
  from matplotlib.legend_handler import HandlerLine2D
  line1, = plt.plot(min_samples_splits, train_results, 'b', label ="Train AUC")
  line2, = plt.plot(min_samples_splits, test_results, 'r', label ="Test AUC")
  plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
  plt.ylabel('AUC score')
  plt.xlabel('min samples split')
  plt.show()

# min_samples_leaf
# min_samples_leaf is The minimum number of samples required to be at a leaf node.
# This similar to min_samples_splits, however, this describe the minimum number of samples of samples at the leafs.
def tuneMinSampleLeaf(x_train, x_test, y_train, y_test):
  min_samples_leafs = np.linspace(0.1, 0.5, 5, endpoint=True)
  train_results = []
  test_results = []
  for min_samples_leaf in min_samples_leafs:
    model = GradientBoostingClassifier(min_samples_leaf=min_samples_leaf)
    model.fit(x_train, y_train)
    train_pred = model.predict(x_train)
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, train_pred)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    train_results.append(roc_auc)
    y_pred = model.predict(x_test)
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    test_results.append(roc_auc)
  from matplotlib.legend_handler import HandlerLine2D
  line1, = plt.plot(min_samples_leafs, train_results, 'b', label ="Train AUC")
  line2, = plt.plot(min_samples_leafs, test_results, 'r', label ="Test AUC")
  plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
  plt.ylabel('AUC score')
  plt.xlabel('min samples leaf')
  plt.show()

# Increasing max_features to consider all of the features results in an overfitting in this case.
# Using max_features = 6 seems to get us the optimal performance.
# The inDepth series investigates how model parameters affect performance in term of overfitting and underfitting
def tuneMaxFeatures(x_train, x_test, y_train, y_test):
  max_features = list(range(1, x_train.shape[1]))
  train_results = []
  test_results = []
  for max_feature in max_features:
    model = GradientBoostingClassifier(max_features=max_feature)
    model.fit(x_train, y_train)
    train_pred = model.predict(x_train)
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, train_pred)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    train_results.append(roc_auc)
    y_pred = model.predict(x_test)
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    test_results.append(roc_auc)

  line1, = plt.plot(max_features, train_results, 'b', label ="Train AUC")
  line2, = plt.plot(max_features, test_results, 'r', label ="Test AUC")
  plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
  plt.ylabel('AUC score')
  plt.xlabel('max features')
  plt.show()

# https://medium.com/all-things-ai/in-depth-parameter-tuning-for-gradient-boosting-3363992e9bae
if __name__ == "__main__":
  scorer = make_scorer(calculate_metrics, greater_is_better=False)
  X_train, X_test, y_train, y_test = get_data_tuple()
  tuneLearnRate(X_train, X_test, y_train, y_test)