#!/usr/bin/env python

# https://towardsdatascience.com/clustering-documents-with-python-97314ad6a78d

import pandas as pd
import wikipedia
import hashlib
import os

titles = ['Data science', 'Artificial intelligence', 'European Central Bank', 'Machine learning', 'World Bank', 'Financial technology', 'International Monetary Fund', 'Basketball', 'Baseball', 'Swimming']
wiki_lst = []
title_lst = []
for title in titles:
    print("loading content: ", title)
    file_name = hashlib.md5(title.encode()).hexdigest()[:7] + ".txt"
    if os.path.exists(file_name):
        with open(file_name, 'r', encoding='utf-8') as infile:
            content = infile.read()
    else:
        content = wikipedia.page(title, auto_suggest=False).content
        with open(file_name, 'w', encoding='utf-8') as outfile:
            outfile.write(content)
    wiki_lst.append(content)
    title_lst.append(title)

print("examine content")
# print(title_lst)
# print(wiki_lst)

from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(stop_words={'english'})
vector = vectorizer.fit_transform(wiki_lst)
# print(X)

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

sum_of_squared_distances = []
K = range(2, 10)

for k in K:
    km = KMeans(n_clusters=k, max_iter=200, n_init=10)
    km = km.fit(vector)
    sum_of_squared_distances.append(km.inertia_)

plt.plot(K, sum_of_squared_distances, 'bx-')
plt.xlabel('k')
plt.ylabel('sum_of_squared_distances')
plt.title('Elbow Method For Optimal k')
plt.show()

true_k = 6
model = KMeans(n_clusters=true_k, init='k-means++', max_iter=200, n_init=10)
model.fit(vector)
labels = model.labels_
wiki_cl = pd.DataFrame(list(zip(title_lst, labels)), columns=['title', 'cluster'])
print(wiki_cl.sort_values(by=['cluster']))

from wordcloud import WordCloud

result = {'cluster': labels, 'wiki': wiki_lst}
result = pd.DataFrame(result)
for k in range(0, true_k):
    s = result[result.cluster == k]
    text = s['wiki'].str.cat(sep=' ' )
    text = text.lower()
    text = ' '.join([word for word in text.split()])
    wordcloud = WordCloud(max_font_size=50, max_words=100, background_color="white").generate(text)
    print("\nCluster: {}".format(k))
    print("Titles")
    titles = wiki_cl[wiki_cl.cluster == k]['title']
    print(titles.to_string(index=False))
    plt.figure()
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.show()
