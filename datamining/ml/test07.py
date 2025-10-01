#!/usr/bin/env python

from sklearn.feature_extraction.text import CountVectorizer


vectorizer = CountVectorizer(min_df = 1)
print(vectorizer)

content = ["How to format my hard dist", " Hard disk format problems "]
X = vectorizer.fit_transform(content)
print("X=", X)
feature_names = vectorizer.get_feature_names_out()
print("feature_names=", feature_names)

print(X.toarray().transpose())
# i번째 컬럼은 i번째 문장에 특정 단어의 등장 여부
# ex) format과 hard는 첫번째 문장과 두번째 문장에 모두 등장

