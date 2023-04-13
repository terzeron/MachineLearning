#!/usr/bin/env python

import os
import sys
import nltk.stem
from sklearn.feature_extraction.text import CountVectorizer
import scipy as sp

DIR="data"
posts = [open(os.path.join(DIR, f)).read() for f in os.listdir(DIR)]
print("posts=", posts)

english_stemmer = nltk.stem.SnowballStemmer('english')
class StemmedCountVectorizer(CountVectorizer):
    def build_analyzer(self):
        analyzer = super(StemmedCountVectorizer, self).build_analyzer()
        return lambda doc: (english_stemmer.stem(w) for w in analyzer(doc))

#vectorizer = CountVectorizer(min_df = 1, stop_words = 'english')
vectorizer = StemmedCountVectorizer(min_df = 1, stop_words = 'english')
print(vectorizer)
print("vectorizer.get_stop_words()=", sorted(vectorizer.get_stop_words())[0:20])

X_train = vectorizer.fit_transform(posts)
print("X_train=", X_train)
num_samples, num_features = X_train.shape
print("#samples={0}, #features={1}".format(num_samples, num_features))
feature_names = vectorizer.get_feature_names()
print("feature_names=", feature_names)

new_post = "imaging databases"
new_post_vec = vectorizer.transform([new_post])
print("new_post_vec=", new_post_vec)
print("new_post_vec.toarray()=", new_post_vec.toarray())

def dist_raw(v1, v2):
    delta = v1 - v2
    return sp.linalg.norm(delta.toarray())

def dist_norm(v1, v2):
    v1_norm = v1 / sp.linalg.norm(v1.toarray())
    v2_norm = v2 / sp.linalg.norm(v2.toarray())
    delta = v1_norm - v2_norm
    return sp.linalg.norm(delta.toarray())

best_doc = None
best_dist = sys.maxsize
best_i = None
dist = dist_norm
for i in range(0, num_samples):
    post = posts[i]
    if post == new_post:
        continue
    post_vec = X_train.getrow(i)
    print("post_vec=", post_vec)
    print("post_vec.toarray()=", post_vec.toarray())
    d = dist(post_vec, new_post_vec)
    print("=== Post {0} with dist={1}: {2}".format(i, d, post))
    if d < best_dist:
        best_dist = d
        best_i = i
print("Best post is {0} with dist={1}".format(best_i, best_dist))
