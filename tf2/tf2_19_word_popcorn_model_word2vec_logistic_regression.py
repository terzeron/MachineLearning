#!/usr/bin/env python


import os
import numpy as np
import pandas as pd
import logging
from gensim.models import word2vec
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def get_features(words, model, num_features):
    # 출력 벡터 초기화
    feature_vector = np.zeros((num_features), dtype=np.float32)
    num_words = 0
    # 어휘사전 준비
    index2word_set = set(model.wv.index2word)
    for w in words:
        if w in index2word_set:
            num_words += 1
            # 사전에 해당하는 단어의 임베딩 값을 더함
            feature_vector = np.add(feature_vector, model[w])
    # 문장의 단어 수로 나누어 정규화
    feature_vector = np.divide(feature_vector, num_words)
    return feature_vector


def get_dataset(reviews, model, num_features):
    dataset = list()
    for s in reviews:
        dataset.append(get_features(s, model, num_features))
    reviewFeatureVecs = np.stack(dataset)
    return reviewFeatureVecs


# TF-IDF를 이용해 문장 벡터 생성

DATA_IN_PATH = "word2vec-nlp-tutorial/"
TRAIN_CLEAN_DATA = "train_clean.csv"

train_data = pd.read_csv(DATA_IN_PATH + TRAIN_CLEAN_DATA)
reviews = list(train_data['review'])
sentiments = list(train_data['sentiment'])

# preprocessing

sentences = []
for review in reviews:
    sentences.append(review.split())

# vectorization
num_features = 300
min_word_count = 40
num_workers = 4
context = 10
downsampling = 1e-3
print("Training model...")
model = word2vec.Word2Vec(sentences, workers=num_workers, vector_size=num_features, min_count=min_word_count, window=context, sample=downsampling)
model_name = "300features_40minwords_10context"
model.save(model_name)

test_data_vecs = get_dataset(sentences, model, num_features)

# split train data & test data

RANDOM_SEED = 42
TEST_SPLIT = 0.2

X = test_data_vecs
y = np.array(sentiments)
X_train, X_eval, y_train, y_eval = train_test_split(X, y, test_size=TEST_SPLIT, random_state=RANDOM_SEED)

lgs = LogisticRegression(class_weight='balanced')
lgs.fit(X_train, y_train)
print("Accuracy: %f" % lgs.score(X_eval, y_eval))

TEST_CLEAN_DATA = "test_clean.csv"
test_data = pd.read_csv(DATA_IN_PATH + TEST_CLEAN_DATA)
print("test_data=", test_data)
test_review = list(test_data['review'])

test_sentences = []
for review in test_review:
    test_sentences.append(review.split())
test_data_vecs = get_dataset(test_sentences, model, num_features)
print("test_data_vecs=", test_data_vecs)

DATA_OUT_PATH = "./data_out/"
test_predicted = lgs.predict(test_data_vecs)
print("test_predicted=", test_predicted)
if not os.path.exists(DATA_OUT_PATH):
    os.makedirs(DATA_OUT_PATH)
ids = list(test_data['id'])
answer_dataset = pd.DataFrame({'id': ids, 'sentiment': test_predicted})
answer_dataset.to_csv(DATA_OUT_PATH + 'lgs_w2v_answer.csv', index=False, quoting=3)
