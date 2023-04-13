#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import numpy
import math

def tfidf(term, doc, docset):
    # tf: 단어의 출현 확률
    tf = float(doc.count(term)) / (sum(doc.count(w) for w in set(doc)))
    print("tf=", tf)
    # idf: 전체 문서 중 특정 단어를 포함한 문서 비율의 역수
    idf = numpy.lib.scimath.log(float(len(docset)) / (len([doc for doc in docset if term in doc])))
    print("idf=", idf)
    return tf * idf

a, abb, abc = ["a"], ["a", "b", "b"], ["a", "b", "c"]
D = [a, abb, abc]
print("tfidf('a', a, D)=", tfidf("a", a, D))
print("tfidf('b', abb, D)=", tfidf("b", abb, D))
print("tfidf('a', abc, D)=", tfidf("a", abc, D))
print("tfidf('b', abc, D)=", tfidf("b", abc, D))
print("tfidf('c', abc, D)=", tfidf("c", abc, D))
