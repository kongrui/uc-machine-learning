#!/usr/bin/env python

import os
import time

import seaborn as sns
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

WORK_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
DATA_DIR = os.path.realpath(WORK_DIR + '/data')
IMG_DIR = os.path.realpath(WORK_DIR + '/images')

NUM_TRAIN_DATA = 74067

def rawDataAnalysis():
    df_train = pd.read_csv(DATA_DIR + "/train.csv.gz", encoding="ISO-8859-1")
    # check the decoration
    print(df_train.columns)
    """
    Index([u'id', u'product_uid', u'product_title', u'search_term', u'relevance'], dtype='object')
    """
    print(df_train['relevance'].describe())
    """
    count    74067.000000
    mean         2.381634
    std          0.533984
    min          1.000000
    25%          2.000000
    50%          2.330000
    75%          3.000000
    max          3.000000
    The relevance is a number between 1 to 3. 
     1 (not relevant)
     2 (mildly relevant)
     3 (highly relevant). 
    the query and products are mostly relevance per countplot 
    """
    print(df_train['search_term'].describe())
    """
    unique                   11795
    top       bed frames headboaed 16
    """
    sns.distplot(df_train['relevance'])
    plt.savefig(IMG_DIR + '/01.relevance-dist-plot.png',)
    plt.show()
    sns.countplot(x="relevance", data=df_train)
    plt.savefig(IMG_DIR + '/01.relevance-counter-plot.png')
    plt.show()

if __name__ == '__main__':
    rawDataAnalysis()

