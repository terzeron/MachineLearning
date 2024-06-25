#!/usr/bin/env python

import os
import sys
import math
from collections import defaultdict

def ngrams(string, n):
    """Generate n-grams from the given string."""
    return [string[i:i+n] for i in range(len(string) - n + 1)]

# 인자로 전달된 디렉토리의 파일 리스트를 가져옵니다.
if len(sys.argv) < 2:
    print("Please provide a directory as an argument.")
    exit()

directory = sys.argv[1]

if not os.path.isdir(directory):
    print(f"'{directory}' is not a valid directory.")
    exit()

#files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
files = [os.path.splitext(f)[0] for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]

# 파일 이름을 기반으로 그룹화합니다.
clusters = defaultdict(set)
scores = {}

max_ngram_length = max(len(f) for f in files)
for n in range(max_ngram_length, 2, -1):  # 2-gram부터 시작
    for f in files:
        for gram in ngrams(f, n):
            clusters[gram].add(f)

# 클러스터링 점수를 계산합니다: n-gram의 길이 * 클러스터 크기
for gram, grouped_files in clusters.items():
    scores[gram] = len(grouped_files) * len(gram)

# 점수가 높은 상위 N개의 클러스터를 출력합니다. 여기서는 상위 10개를 출력하도록 설정했습니다.
top_clusters = sorted(scores, key=scores.get, reverse=True)[:10]

for gram in top_clusters:
    print(f"Cluster (based on n-gram '{gram}', score: {scores[gram]}): {', '.join(clusters[gram])}")

