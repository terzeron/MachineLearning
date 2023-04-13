#!/usr/bin/env python

# https://stackoverflow.com/questions/28160335/plot-a-document-tfidf-2d-graph

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

newsgroups_train = fetch_20newsgroups(subset='train', categories=['alt.atheism', 'sci.space', 'talk.religion.misc', 'comp.graphics'])
#print(newsgroups_train)
print(newsgroups_train.DESCR)

pipeline = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
])
X = pipeline.fit_transform(newsgroups_train.data).todense()
#print(X)
print("X.shape=", X.shape)

pca = PCA(n_components=2).fit(X)
data2D = pca.transform(X)
#print(data2D)
print("data2D.shape=", data2D.shape)
# plt.scatter(data2D[:,0], data2D[:,1], c=newsgroups_train.target)
# plt.show()

## Nearest neighbour
kmeans = KMeans(n_clusters=2).fit(X)
centers2D = pca.transform(kmeans.cluster_centers_)

# plt.hold(True)
plt.scatter(data2D[:, 0], data2D[:, 1], c=newsgroups_train.target)
plt.scatter(centers2D[:, 0], centers2D[:, 1],
            marker='x', s=200, linewidths=3, c='r')
plt.show()
