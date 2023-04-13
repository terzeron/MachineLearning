#!/usr/bin/env python

# https://www.lucypark.kr/courses/2015-ba/text-mining.html

import pprint

from konlpy.corpus import kobill    # Docs from pokr.kr/bill
files_ko = kobill.fileids()         # Get file ids
doc_ko = kobill.open('1809890.txt').read()
#print(doc_ko)

from konlpy.tag import Okt; 
t = Okt()
tokens_ko = t.morphs(doc_ko)
#print(tokens_ko)

import nltk
ko = nltk.Text(tokens_ko, name='대한민국 국회 의안 제 1809890호')   # For Python 2, input `name` as u'유니코드'
#print(ko)

print("total ko.tokens:", len(ko.tokens))       # returns number of tokens (document length)
print("unique ko.tokens:", len(set(ko.tokens)))  # returns number of unique tokens
vocab = ko.vocab()                  # returns frequency distribution
print("vocab:", end=" ")
pprint.pprint(vocab)
ko.plot(50)     # Plot sorted frequency of top 50 tokens

from matplotlib import pylab, rc
pylab.show = lambda: pylab.savefig('some_filename.png')
rc('font', family='NanumGothic')

count = ko.count('초등학교')
print("count:", count)
ko.dispersion_plot(['육아휴직', '초등학교', '공무원'])
concordance = ko.concordance('초등학교')
print("concordance:", concordance)
similar = ko.similar('자녀')
print("similar:", similar)
similar = ko.similar('육아휴직')
print("similar:", similar)
collocations = ko.collocations()
print("collocations:", collocations)

#
# Tagging and chunking
#

from konlpy.tag import Okt;
t = Okt()
tags_ko = t.pos("작고 노란 강아지가 페르시안 고양이에게 짖었다")
print("tags_ko:", tags_ko)

parser_ko = nltk.RegexpParser("NP: {<Adjective>*<Noun>*}")
chunks_ko = parser_ko.parse(tags_ko)
print("chunks_ko:", chunks_ko)
#chunks_ko.draw()

#
# Topic Modeling
#

# preprocessing
from konlpy.corpus import kobill
docs_ko = [kobill.open(i).read() for i in kobill.fileids()]

from konlpy.tag import Okt;
t = Okt()
pos = lambda d: ['/'.join(p) for p in t.pos(d, stem=True, norm=True)]
texts_ko = [pos(doc) for doc in docs_ko]
print(texts_ko[0])

from gensim import corpora
dictionary_ko = corpora.Dictionary(texts_ko)
dictionary_ko.save('ko.dict')  # save dictionary to file for future use

from gensim import models
tf_ko = [dictionary_ko.doc2bow(text) for text in texts_ko]
tfidf_model_ko = models.TfidfModel(tf_ko)
tfidf_ko = tfidf_model_ko[tf_ko]
corpora.MmCorpus.serialize('ko.mm', tfidf_ko) # save corpus to file for future use

# print first 10 elements of first document's tf-idf vector
print("corpus:", tfidf_ko.corpus[0][:10])
# print top 10 elements of first document's tf-idf vector
print("first column of corpus:", sorted(tfidf_ko.corpus[0], key=lambda x: x[1], reverse=True)[:10])
# print token of most frequent element
print("dictionary:", dictionary_ko.get(414))

# train

# LSI
ntopics, nwords = 3, 5
lsi_ko = models.lsimodel.LsiModel(tfidf_ko, id2word=dictionary_ko, num_topics=ntopics)
print("lsi:topics:", lsi_ko.print_topics(num_topics=ntopics, num_words=nwords))

# LDA
import numpy as np; np.random.seed(42)  # optional
lda_ko = models.ldamodel.LdaModel(tfidf_ko, id2word=dictionary_ko, num_topics=ntopics)
print("lda:topics:", lda_ko.print_topics(num_topics=ntopics, num_words=nwords))

# HDP
import numpy as np; np.random.seed(42)  # optional
hdp_ko = models.hdpmodel.HdpModel(tfidf_ko, id2word=dictionary_ko)
print("hdp:topics:", hdp_ko.print_topics(num_topics=ntopics, num_words=nwords))

bow = tfidf_model_ko[dictionary_ko.doc2bow(texts_ko[0])]
sorted(lsi_ko[bow], key=lambda x: x[1], reverse=True)
sorted(lda_ko[bow], key=lambda x: x[1], reverse=True)
sorted(hdp_ko[bow], key=lambda x: x[1], reverse=True)

bow = tfidf_model_ko[dictionary_ko.doc2bow(texts_ko[1])]
sorted(lsi_ko[bow], key=lambda x: x[1], reverse=True)
sorted(lda_ko[bow], key=lambda x: x[1], reverse=True)
sorted(hdp_ko[bow], key=lambda x: x[1], reverse=True)

#
# Word embedding
#

# word2vec
from konlpy.corpus import kobill
docs_ko = [kobill.open(i).read() for i in kobill.fileids()]

# tokenize
from konlpy.tag import Okt;
t = Okt()
pos = lambda d: ['/'.join(p) for p in t.pos(d)]
texts_ko = [pos(doc) for doc in docs_ko]

# train
from gensim.models import word2vec
wv_model_ko = word2vec.Word2Vec(texts_ko)
wv_model_ko.wv.init_sims(replace=True)
#wv_model_ko.wv.fill_norms()
wv_model_ko.wv.save('ko_word2vec.model')

# test
most_similar1 = wv_model_ko.wv.most_similar(pos('정부'), topn=20)
print("most_similar1:", most_similar1)
most_similar2 = wv_model_ko.wv.most_similar(pos('초등학교'), topn=20)
print("most_similar2:", most_similar2)

