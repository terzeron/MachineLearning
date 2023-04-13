#!/usr/bin/env python

# https://hoonzi-text.tistory.com/19

import pandas as pd
from konlpy.tag import Okt
from matplotlib import pyplot as plt
from tqdm import tqdm

df = pd.read_pickle("daum_news.pkl")

okt = Okt()  # 형태소 분석기 객체 생성
noun_list = []
for content in tqdm(df['content']):
    nouns = okt.nouns(content)  # 명사만 추출하기, 결과값은 명사 리스트
    noun_list.append(nouns)
df['nouns'] = noun_list
# df.head()
print(df['nouns'])

drop_index_list = []  # 지워버릴 index를 담는 리스트
for i, row in df.iterrows():
    temp_nouns = row['nouns']
    if len(temp_nouns) == 0:  # 만약 명사리스트가 비어 있다면
        drop_index_list.append(i)  # 지울 index 추가
df = df.drop(drop_index_list)  # 해당 index를 지우기
# index를 지우면 순회시 index 값이 중간중간 비기 때문에 index를 다시 지정
df.index = range(len(df))

from sklearn.feature_extraction.text import TfidfVectorizer

# 문서를 명사 집합으로 보고 문서 리스트로 치환 (tfidfVectorizer 인풋 형태를 맞추기 위해)
text = [" ".join(noun) for noun in df['nouns']]
# print(text)
# 전체 문서중 단어 빈도가 5미만 일경우 단어 계산에서 제외
# min_df: 해당 단어가 출현하는 문서의 최소 수
# ngram_range: 단어 몇개까지를 하나의 단어로 볼건지에 대한 정보, A, AB, ABC, ABCD까지 한 단어
tfidf_vectorizer = TfidfVectorizer(min_df=5, ngram_range=(2, 3))
tfidf_vectorizer.fit(text)
vector = tfidf_vectorizer.transform(text).toarray()
print(vector)

from sklearn.cluster import DBSCAN
import numpy as np

vector = np.array(vector)  # Normalizer를 이용해 변환된 벡터
# 밀도 기반의 클러스터링, 일정 기준치 이상의 점들이 존재하면 클러스터링, 기준점을 옮김, 노이즈에 강함. 속도가 느림
# eps: eps만큼의 거리 안의 모든 포인트를 포함시킴
# min_samples: eps만큼의 거리 안의 포인트 수가 min_samples보다 작으면 노이즈로 처리
model = DBSCAN(eps=0.5, min_samples=5, metric="cosine")
# 거리 계산 식으로는 Cosine distance를 이용
result = model.fit_predict(vector)
df['result'] = result
# df.head()
print(df['result'])

for cluster_num in set(result):
    # -1,0은 노이즈 판별이 났거나 클러스터링이 안된 경우
    if (cluster_num == -1 or cluster_num == 0):
        continue
    else:
        print("cluster num : {}".format(cluster_num))
        temp_df = df[df['result'] == cluster_num]  # cluster num 별로 조회
        for title in temp_df['title']:
            print(title)  # 제목으로 살펴보자
        print()
