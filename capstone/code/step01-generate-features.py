#!/usr/bin/env python

import os
import time

import numpy as np
import pandas as pd

from nltk.stem.porter import *
STEMMER = PorterStemmer()

WORK_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
DATA_DIR = os.path.realpath(WORK_DIR + '/data')
OUT_DIR = os.path.realpath(WORK_DIR + '/features')

def str_norm(str1, stemmer=STEMMER):
    str1 = str1.lower()
    str1 = str1.replace(" in.","in.")
    str1 = str1.replace(" inch","in.")
    str1 = str1.replace("inch","in.")
    str1 = str1.replace(" in ","in. ")
    str1 = (" ").join([stemmer.stem(z) for z in str1.split(" ")])
    return str1

def str_common_word(str1, str2):
    words, cnt = str1.split(), 0
    for word in words:
        if str2.find(word)>=0:
            cnt+=1
    return cnt

def str_whole_word(str1, str2):
    str1, str2 = str1.strip(), str2.strip()
    cnt = 0
    i_ = 0
    while i_ < len(str2):
        i_ = str2.find(str1, i_)
        if i_ == -1:
            return cnt
        else:
            cnt += 1
            i_ += len(str1)
    return cnt

def generateFeatures():
    ROW_HEAD = 10
    time_start = time.time()

    df_train = pd.read_csv(DATA_DIR + '/train.csv.gz', encoding="ISO-8859-1")
    num_train = df_train.shape[0]
    print("Training Data Size : %d " % num_train)
    df_test = pd.read_csv(DATA_DIR + '/test.csv.gz', encoding="ISO-8859-1")
    df_prodDesc = pd.read_csv(DATA_DIR + '/product_descriptions.csv.gz')

    df_all = pd.concat((df_train, df_test), axis=0, ignore_index=True)
    print(df_all.head(ROW_HEAD))
    df_all = pd.merge(df_all, df_prodDesc, how='left', on='product_uid')

    print("No of rows : %d " % df_all.shape[0]) # 240,760

    # Generate sample set
    # ROW_SMALL_SET = 1000
    # df_all = df_all.head(ROW_SMALL_SET)

    # norm the text - lower, stemming
    # remove stop words and special chars
    # norm on numbers and brands ...
    df_all['search_term'] = df_all['search_term'].map(lambda x: str_norm(x))
    df_all['product_title'] = df_all['product_title'].map(lambda x: str_norm(x))
    df_all['product_description'] = df_all['product_description'].map(lambda x: str_norm(x))

    time_norm = time.time()
    print("Normalizing... dur=%s" % round(((time_norm - time_start) / 60), 2))

    # derive text feature
    df_all['len_query'] = df_all['search_term'].map(lambda x: len(x.split())).astype(np.int64)
    df_all['len_title'] = df_all['product_title'].map(lambda x: len(x.split())).astype(np.int64)
    df_all['len_desc'] = df_all['product_description'].map(lambda x: len(x.split())).astype(np.int64)
    df_all['query_feq_title'] = df_all.apply(lambda x: str_whole_word(x['search_term'], x['product_title']), axis=1)
    df_all['query_feq_desc'] = df_all.apply(lambda x: str_whole_word(x['search_term'], x['product_description']), axis=1)
    df_all['term_feq_title'] = df_all.apply(lambda x: str_common_word(x['search_term'], x['product_title']), axis=1)
    df_all['term_feq_desc'] = df_all.apply(lambda x: str_common_word(x['search_term'], x['product_description']), axis=1)
    df_all['term_ratio_title'] = df_all['term_feq_title'] / df_all['len_query']
    df_all['term_ratio_desc'] = df_all['term_feq_desc'] / df_all['len_query']

    print(df_all.head(ROW_HEAD))

    df_all.drop(['search_term', 'product_title', 'product_description', 'product_uid'], axis=1, inplace=True)

    df_all.to_csv(OUT_DIR + "/data.csv.gz", compression='gzip', index=False)

    time_end = time.time()
    print("--- Norm: %s minutes ---" % round(((time_norm - time_start) / 60), 2))
    print("--- Took: %s minutes ---" % round(((time_end- time_start) / 60), 2))
    print("Training Data Size : %d " % num_train)
    # --- Norm: 27.67 minutes ---
    # --- Took: 28.34 minutes ---
    # num_train = 74067

if __name__ == '__main__':
    generateFeatures()

