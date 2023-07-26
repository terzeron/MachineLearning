#!/usr/bin/env python
import math

from sklearn.feature_extraction.text import TfidfVectorizer
sent = ("휴일 인 오늘 도 서쪽 을 중심 으로 폭염 이 이어졌는데요, 내일 은 반가운 비 소식 이 있습니다.", "폭염 을 피해서 휴일 에 놀러왔다가 갑작스런 비 로 인해 망연자실 하고 있습니다.")
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(sent)

idf = tfidf_vectorizer.idf_
print(dict(zip(tfidf_vectorizer.get_feature_names_out(), idf)))

from sklearn.metrics.pairwise import cosine_similarity
print(cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2]))

# non-normalized vectors
from sklearn.metrics.pairwise import euclidean_distances
print(euclidean_distances(tfidf_matrix[0:1], tfidf_matrix[1:2]))

# normalized vectors
import numpy as np
def l1_normalize(v):
    norm = np.sum(v)
    return v/norm
tfidf_norm_l1 = l1_normalize(tfidf_matrix)
print(euclidean_distances(tfidf_norm_l1[0:1], tfidf_norm_l1[1:2]))

from sklearn.metrics.pairwise import manhattan_distances
print(manhattan_distances(tfidf_matrix[0:1], tfidf_matrix[1:2]))
# semi-normalized
print(manhattan_distances(tfidf_norm_l1[0:1], tfidf_norm_l1[1:2])/math.sqrt(2))

