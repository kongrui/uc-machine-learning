#!/usr/bin/env python

import os
import time

import seaborn as sns
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

WORK_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
DATA_DIR = os.path.realpath(WORK_DIR + '/features')
TEMP_DIR = os.path.realpath(WORK_DIR + '/temp')

NUM_TRAIN_DATA = 74067

def loadData():
    data = pd.read_csv(DATA_DIR + '/data.csv.gz', encoding="ISO-8859-1").iloc[:NUM_TRAIN_DATA]
    #print(data.head(5))
    #print(data.describe())
    #print(data.info)
    return data

def histogram(data):
    # histogram
    data.hist(bins=50, figsize=(20,15))
    plt.savefig(TEMP_DIR + '/histogram.png',)
    plt.show()



#
# Split products into buckets based on their product_uid
# the relevance scores for products with different product ids
#
def pointplot(data):
    data["bucket"] = np.floor(data["product_uid"] / 1000)
    sns.pointplot(x="bucket", y="relevance", data=data[["bucket", "relevance"]])
    plt.savefig(TEMP_DIR + '/pointplot.png',)
    plt.show()


def corrmatrix(data):
    # coorelation
    corr_matrix = data.corr()
    print(corr_matrix['relevance'].sort_values(ascending=False))
    """
    relevance           1.000000
    term_ratio_title    0.352815
    term_ratio_desc     0.285134
    term_feq_title      0.215921
    query_feq_title     0.170965
    term_feq_desc       0.161321
    query_feq_desc      0.086555
    len_desc            0.040001
    len_title          -0.019840
    len_query          -0.073189
    """

    #df_all['term_ratio_title'] = df_all['term_feq_title'] / df_all['len_query']
    #df_all['term_ratio_desc'] = df_all['term_feq_desc'] / df_all['len_query']


def exploreFeatures():
    data = loadData()
    #corrmatrix(data)
    #countplot(data)
    pointplot(data)

    #target = data['relevance']

    #features = data.drop('relevance', axis=1)
    #sns.pairplot(features.iloc[:, :])
    #sns.regplot(x="term_few_desc", y="relevance", data=data)

if __name__ == '__main__':
    exploreFeatures()

