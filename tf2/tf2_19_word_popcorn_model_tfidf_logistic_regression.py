#!/usr/bin/env python


import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score, precision_score

# TF-IDF를 이용해 문장 벡터 생성

DATA_IN_PATH = "word2vec-nlp-tutorial/"
TRAIN_CLEAN_DATA = "train_clean.csv"
train_data = pd.read_csv(DATA_IN_PATH + TRAIN_CLEAN_DATA)
reviews = list(train_data['review'])
sentiments = list(train_data['sentiment'])

vectorizer = TfidfVectorizer(min_df=0.0, analyzer="char", sublinear_tf=True, ngram_range=(1, 3), max_features=5000)
X = vectorizer.fit_transform(reviews)

# split

RANDOM_SEED = 42
TEST_SPLIT = 0.2

y = np.array(sentiments)
X_train, X_eval, Y_train, Y_eval = train_test_split(X, y, test_size=TEST_SPLIT, random_state=RANDOM_SEED)

# train

lgs = LogisticRegression(class_weight='balanced')
lgs.fit(X_train, Y_train)
print("Accuracy: %f" % lgs.score(X_eval, Y_eval))
print("Recall: %f" % recall_score(Y_eval, lgs.predict(X_eval)))
print("Precision: %f" % precision_score(Y_eval, lgs.predict(X_eval)))

# test

TEST_CLEAN_DATA = "test_clean.csv"
test_data = pd.read_csv(DATA_IN_PATH + TEST_CLEAN_DATA)
print("test_data=", test_data)
testDataVecs = vectorizer.transform(test_data['review'])
test_predicted = lgs.predict(testDataVecs)
print("test_predicted=", test_predicted)

DATA_OUT_PATH = './data_out/'
if not os.path.exists(DATA_OUT_PATH):
    os.makedirs(DATA_OUT_PATH)
ids = list(test_data['id'])
answer_dataset = pd.DataFrame({'id': ids, 'sentiment': test_predicted})
answer_dataset.to_csv(DATA_OUT_PATH + 'lgs_tfidf_answer.csv', index=False, quoting=3)
