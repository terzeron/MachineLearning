#!/usr/bin/env python

from sklearn.feature_extraction.text import CountVectorizer

text_data = ['나는 배가 고프다', '내일 점심 뭐먹지', '내일 공부 해야겠다', '점심 먹고 공부해야지']
count_vectorizer = CountVectorizer()
count_vectorizer.fit(text_data)
print("vocabulary=", count_vectorizer.vocabulary_)
print("len(vocabulary=", len(count_vectorizer.vocabulary_))

sentence = [text_data[0]]
print("sentence=", sentence)
print("vectorizer.transform(sentence)=", count_vectorizer.transform(sentence).toarray())




