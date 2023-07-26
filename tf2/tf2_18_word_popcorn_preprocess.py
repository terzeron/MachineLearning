#!/usr/bin/env python

import numpy as np
import pandas as pd
import os
import re
import matplotlib.pyplot as plt
import seaborn as sns
#%matplotlib inline

DATA_IN_PATH = "word2vec-nlp-tutorial/"
train_data = pd.read_csv(DATA_IN_PATH + "labeledTrainData.tsv", header=0, delimiter="\t", quoting=3)
print(train_data.head())

print("파일 크기 : ")
for file in os.listdir(DATA_IN_PATH):
    if 'tsv' in file and 'zip' not in file:
        print(file.ljust(30) + str(round(os.path.getsize(DATA_IN_PATH + file) / 1000000, 2)) + "MB")

print("전체 학습 데이터의 개수: {}".format(len(train_data)))
train_length = train_data['review'].apply(len)
print(train_length.head())

plt.figure(figsize=(12, 5))
plt.hist(train_length, bins=200, alpha=0.5, color='r', label='word')
plt.yscale('log')
plt.title('Log-Histogram of length of review')
plt.xlabel('Length of review')
plt.ylabel('Number of reviews')
plt.show()

train_word_counts = train_data['review'].apply(lambda x: len(x.split(' ')))
plt.figure(figsize=(15, 10))
plt.hist(train_word_counts, bins=50, facecolor='r', label='train')
plt.title('Log-Histogram of word count in review', fontsize=15)
plt.yscale('log')
plt.legend()
plt.xlabel('Number of words', fontsize=15)
plt.ylabel('Number of reviews', fontsize=15)
plt.show()

# 물음표와 마침표가 구두점으로 사용된 경우와 대소문자 비율
qmarks = np.mean(train_data['review'].apply(lambda x: '?' in x))
fullstop = np.mean(train_data['review'].apply(lambda x: '.' in x))
capital_first = np.mean(train_data['review'].apply(lambda x: x[0].isupper()))
capitals = np.mean(train_data['review'].apply(lambda x: max([y.isupper() for y in x])))
numbers = np.mean(train_data['review'].apply(lambda x: max([y.isdigit() for y in x])))

print("물음표가 있는 질문: {:.2f}%".format(qmarks * 100))
print("마침표가 있는 질문: {:.2f}%".format(fullstop * 100))
print("첫 글자가 대문자인 질문: {:.2f}%".format(capital_first * 100))
print("대문자가 있는 질문: {:.2f}%".format(capitals * 100))
print("숫자가 있는 질문: {:.2f}%".format(numbers * 100))

# 전처리
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer

print(train_data['review'][0])

# html 태그, 특수문자 제거
def preprocessing(review, remove_stopwords=False):
    review_text = BeautifulSoup(review, 'html5lib').get_text()
    review_text = review_text.replace('()', ' ')
    review_text = re.sub('[^a-zA-Z]', ' ', review_text)
    words = review_text.lower().split()
    if remove_stopwords:
        stops = set(stopwords.words('english'))
        words = [w for w in words if not w in stops]

    clean_review = ' '.join(words)
    return clean_review

clean_train_reviews = []
for review in train_data['review']:
    clean_train_reviews.append(preprocessing(review, remove_stopwords=True))
print(clean_train_reviews[0])

clean_train_df = pd.DataFrame({'review': clean_train_reviews, 'sentiment': train_data['sentiment']})
tokenizer = Tokenizer()
tokenizer.fit_on_texts(clean_train_reviews)
# text_sequences는 인덱스만으로 구성됨
text_sequences = tokenizer.texts_to_sequences(clean_train_reviews)
print(text_sequences[0])

# word_index를 이용해서 vocabulary 구축
word_vocab = tokenizer.word_index
word_vocab["<PAD>"] = 0
#print(word_vocab)
print("전체 단어 개수: ", len(word_vocab))

data_configs = {}
data_configs["vocab"] = word_vocab
data_configs["vocab_size"] = len(word_vocab)

# 데이터 길이 정규화
MAX_SEQUENCE_LENGTH = 174
train_inputs = pad_sequences(text_sequences, maxlen=MAX_SEQUENCE_LENGTH, padding='post')
print("shape of train data: ", train_inputs.shape)

train_labels = np.array(train_data['sentiment'])
print("Shape of label tensor: ", train_labels.shape)

# 학습에 사용할 전처리 데이터 저장

TRAIN_INPUT_DATA = "train_input.npy"
TRAIN_LABEL_DATA = "train_label.npy"
TRAIN_CLEAN_DATA = "train_clean.csv"
DATA_CONFIGS = "data_configs.json"

import os
import json
if not os.path.exists(DATA_IN_PATH):
    os.makedirs(DATA_IN_PATH)

np.save(open(DATA_IN_PATH + TRAIN_INPUT_DATA, 'wb'), train_inputs)
np.save(open(DATA_IN_PATH + TRAIN_LABEL_DATA, 'wb'), train_labels)
clean_train_df.to_csv(DATA_IN_PATH + TRAIN_CLEAN_DATA, index=False)
json.dump(data_configs, open(DATA_IN_PATH + DATA_CONFIGS, 'w'), ensure_ascii=False)

# 평가 데이터 저장

test_data = pd.read_csv(DATA_IN_PATH + "testData.tsv", header=0, delimiter="\t", quoting=3)
clean_test_reivews = []
for review in test_data['review']:
    clean_test_reivews.append(preprocessing(review, remove_stopwords=True))
clean_test_df = pd.DataFrame({'review': clean_test_reivews, 'id': test_data['id']})
test_id = np.array(test_data['id'])

text_sequences = tokenizer.texts_to_sequences(clean_test_reivews)
test_inputs = pad_sequences(text_sequences, maxlen=MAX_SEQUENCE_LENGTH, padding='post')

TEST_INPUT_DATA = "test_input.npy"
TEST_CLEAN_DATA = "test_clean.csv"
TEST_ID_DATA = "test_id.npy"

np.save(open(DATA_IN_PATH + TEST_INPUT_DATA, 'wb'), test_inputs)
np.save(open(DATA_IN_PATH + TEST_ID_DATA, 'wb'), test_id)
clean_test_df.to_csv(DATA_IN_PATH + TEST_CLEAN_DATA, index=False)

