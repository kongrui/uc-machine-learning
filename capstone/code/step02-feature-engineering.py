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

def str_norm(strIn, source="U", stemmer=STEMMER, debug=False):
    str1 = strIn
    str1 = str1.lower()
    str1 = str_standardize_units(str1)
    strOut = (" ").join([stemmer.stem(z) for z in str1.split(" ")])
    if debug and strIn != strOut:
        print ("\n" + source + "#IN #" + strIn + '\n' + source + "#OUT#" + strOut)
    return strOut

def str_standardize_units(s, replacements={"'|in|inches|inch": "in",
                                           "''|ft|foot|feet": "ft",
                                           "pound|pounds|lb|lbs": "lb",
                                           "volt|volts|v": "v",
                                           "watt|watts|w": "w",
                                           "ounce|ounces|oz": "oz",
                                           "gal|gallon": "gal",
                                           "m|meter|meters": "m",
                                           "cm|centimeter|centimeters": "cm",
                                           "mm|milimeter|milimeters": "mm",
                                           "yd|yard|yards": "yd",
                                           }):
    # Add spaces after measures
    regexp_template = r"([/\.0-9]+)[-\s]*({0})([,\.\s]|$)"
    regexp_subst_template = "\g<1> {0} "

    s = re.sub(r"([^\s-])x([0-9]+)", "\g<1> x \g<2>", s).strip()

    # Standartize unit names
    for pattern, repl in replacements.iteritems():
        s = re.sub(regexp_template.format(pattern), regexp_subst_template.format(repl), s)

    s = re.sub(r"\s\s+", " ", s).strip()
    return s

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

# TODOs
#1 typo
#2 unit standard
#3 number standard
#4 words standarf: singular and plural, verb forms and tenses, etc
# brand name
# stop words
# synonyms
# stop words - from sklearn.feature_extraction import text
# # stop_ = list(text.ENGLISH_STOP_WORDS)
def preprocessingTextFeatures(df_all, debug=False):
    df_all['search_term'] = df_all['search_term'].map(lambda x: str_norm(x, source="Q", debug=debug))
    df_all['product_title'] = df_all['product_title'].map(lambda x: str_norm(x, source="T", debug=debug))
    df_all['product_description'] = df_all['product_description'].map(lambda x: str_norm(x, source="D", debug=debug))

#
# brand name
def extractNumberFeatures(df_all):
    df_all['len_query'] = df_all['search_term'].map(lambda x: len(x.split())).astype(np.int64)
    df_all['len_title'] = df_all['product_title'].map(lambda x: len(x.split())).astype(np.int64)
    df_all['len_desc'] = df_all['product_description'].map(lambda x: len(x.split())).astype(np.int64)
    df_all['query_feq_title'] = df_all.apply(lambda x: str_whole_word(x['search_term'], x['product_title']), axis=1)
    df_all['query_feq_desc'] = df_all.apply(lambda x: str_whole_word(x['search_term'], x['product_description']),
                                            axis=1)
    df_all['term_feq_title'] = df_all.apply(lambda x: str_common_word(x['search_term'], x['product_title']), axis=1)
    df_all['term_feq_desc'] = df_all.apply(lambda x: str_common_word(x['search_term'], x['product_description']),
                                           axis=1)
    df_all['term_ratio_title'] = df_all['term_feq_title'] / df_all['len_query']
    df_all['term_ratio_desc'] = df_all['term_feq_desc'] / df_all['len_query']

def featureEngineering(sampleSize=0):
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
    if sampleSize > 0:
        debugPreprocessing = False
        df_all = df_all.head(sampleSize)
    else:
        debugPreprocessing = False

    # preprocessing text fields first
    #
    # norm the text - lower, stemming
    # remove stop words and special chars
    # norm on numbers and brands ...
    preprocessingTextFeatures(df_all, debugPreprocessing)
    time_norm = time.time()
    print("Normalizing... dur=%s" % round(((time_norm - time_start) / 60), 2))

    # derive text feature
    extractNumberFeatures(df_all)
    time_extract = time.time()
    print("Extracting... dur=%s" % round(((time_extract - time_norm) / 60), 2))
    print(df_all.head(ROW_HEAD))

    df_all.drop(['search_term', 'product_title', 'product_description'], axis=1, inplace=True)

    if sampleSize > 0:
        datafile = "/data." + str(sampleSize) + ".csv"
        df_all.to_csv(OUT_DIR + datafile, index=False)
    else:
        datafile = "/data.csv.gz"
        df_all.to_csv(OUT_DIR + datafile, compression='gzip', index=False)

    time_end = time.time()
    print("--- Norm: %s minutes ---" % round(((time_norm - time_start) / 60), 2))
    print("--- Took: %s minutes ---" % round(((time_end- time_start) / 60), 2))
    print("Training Data Size : %d " % num_train)
    # --- Norm: 27.67 minutes ---
    # --- Took: 28.34 minutes ---
    # num_train = 74067

if __name__ == '__main__':
    # generate a small set
    # featureEngineering(100)

    # geneare full set
    featureEngineering()

