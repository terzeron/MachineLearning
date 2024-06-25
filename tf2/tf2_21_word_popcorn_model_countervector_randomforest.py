#!/usr/bin/env python
import os

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import train_test_split

DATA_IN_PATH = "./word2vec-nlp-tutorial/"
TRAIN_CLEAN_DATA = "train_clean.csv"

train_data = pd.read_csv(DATA_IN_PATH + TRAIN_CLEAN_DATA)
reviews = list(train_data['review'])
y = np.array(train_data['sentiment'])

vectorizer = CountVectorizer(analyzer="word", max_features=5000)
train_data_features = vectorizer.fit_transform(reviews)
print("train_data_features=", train_data_features)

# split
TEST_SIZE = 0.2
RANDOM_SEED = 42
train_input, eval_input, train_label, eval_label = train_test_split(train_data_features, y, test_size=TEST_SIZE, random_state=RANDOM_SEED)

# 랜덤 포레스트 분류기에 100개의 의사결정 트리를 사용한다.
forest = RandomForestClassifier(n_estimators=100)
# 단어 묶음을 벡터화한 데이터와 정답 데이터를 가지고 학습을 시작한다.
forest.fit(train_input, train_label)
print("Accuracy: %f" % forest.score(eval_input, eval_label))
print("Precision: %f" % precision_score(eval_label, forest.predict(eval_input)))
print("Recall: %f" % recall_score(eval_label, forest.predict(eval_input)))

TEST_CLEAN_DATA = "test_clean.csv"
DATA_OUT_PATH = "./data_out/"

test_data = pd.read_csv(DATA_IN_PATH + TEST_CLEAN_DATA)
test_reviews = list(test_data['review'])
ids = list(test_data['id'])
test_data_features = vectorizer.transform(test_reviews)
if not os.path.exists(DATA_OUT_PATH):
    os.makedirs(DATA_OUT_PATH)
result = forest.predict(test_data_features)
output = pd.DataFrame(data={"id": ids, "sentiment": result})
output.to_csv(DATA_OUT_PATH + "Bag_of_Words_model.csv", index=False, quoting=3)