#!/usr/bin/env python

import numpy as np
import pandas as pd

from regressionutil import DATA_DIR
from regressionutil import IMG_DIR
from regressionutil import NUM_TRAIN_DATA

import seaborn as sns

from matplotlib import pyplot as plt

def loadData():
    data = pd.read_csv(DATA_DIR + '/data.csv.gz', encoding="ISO-8859-1").iloc[:NUM_TRAIN_DATA]
    return data

def basics(data):
    #print(data.head(5))
    print(data.describe().to_csv(index=False, sep='|'))
    """
id|product_uid|relevance|len_query|len_title|len_desc|query_feq_title|query_feq_desc|term_feq_title|term_feq_desc|term_ratio_title|term_ratio_desc
74067.0|74067.0|74067.0|74067.0|74067.0|74067.0|74067.0|74067.0|74067.0|74067.0|74067.0|74067.0
112385.70922273077|142331.91155305333|2.3816337910270433|3.114301915832962|10.587589614808214|130.53104621491352|0.1102515290210215|0.1579785869550542|1.9847975481658497|1.974226038586685|0.6427411993170024|0.6330562709972568
64016.573650215265|30770.774864426763|0.5339839484172036|1.214578987455432|3.702143143668057|75.69489432917908|0.31573822617413233|0.6808966030753788|1.1424767744362694|1.165310380420707|0.30476523581617465|0.30652030011230785
2.0|100001.0|1.0|1.0|1.0|20.0|0.0|0.0|0.0|0.0|0.0|0.0
57163.5|115128.5|2.0|2.0|8.0|78.0|0.0|0.0|1.0|1.0|0.5|0.5
113228.0|137334.0|2.33|3.0|10.0|111.0|0.0|0.0|2.0|2.0|0.6666666666666666|0.6666666666666666
168275.5|166883.5|3.0|4.0|13.0|161.0|0.0|0.0|3.0|3.0|1.0|1.0
221473.0|206650.0|3.0|11.0|35.0|845.0|2.0|24.0|10.0|11.0|1.0|1.0
    """
    #print(data.info)
    return data

def histogram(data):
    # histogram
    data.hist(bins=50, figsize=(20,15))
    plt.savefig(IMG_DIR + '/02.relevance-histogram.png',)
    #plt.show()

#
# Split products into buckets based on their product_uid
# the relevance scores for products with different product ids
#
def pointplot(data):
    data["bucket"] = np.floor(data["product_uid"] / 1000)
    sns.pointplot(x="bucket", y="relevance", data=data[["bucket", "relevance"]])
    plt.savefig(IMG_DIR + '/02.relevance-productid-pointplot.png')
    #plt.show()

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

def pairplot(data):
    #features = data.drop('relevance', axis=1)
    #sns.pairplot(features.iloc[:, :])
    sns.pairplot(data)
    plt.savefig(IMG_DIR + '/02.pairplot.png')
    #sns.regplot(x="term_few_desc", y="relevance", data=data)


def exploreFeatures():
    data = loadData()
    #basics(data)
    corrmatrix(data)
    #pointplot(data)
    #pairplot(data)

if __name__ == '__main__':
    exploreFeatures()

