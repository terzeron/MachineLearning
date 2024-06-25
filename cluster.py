#!/usr/bin/env python

import os
import sys
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
seen_clusters = set()
MIN_CLUSTER_SIZE = int(sys.argv[2])

# 긴 n-gram부터 시작하여 파일 이름을 그룹화합니다.
max_ngram_length = max(len(f) for f in files)
for n in range(max_ngram_length, 7, -1):  # 2-gram부터 시작
    for f in files:
        for gram in ngrams(f, n):
            clusters[gram].add(f)

# 크기가 MIN_CLUSTER_SIZE보다 큰 클러스터만 출력합니다.
for gram, grouped_files in clusters.items():
    if len(grouped_files) >= MIN_CLUSTER_SIZE:
        # 중복된 클러스터를 제거하기 위해 frozenset으로 클러스터를 표현하고 집합에 저장합니다.
        if frozenset(grouped_files) not in seen_clusters:
            seen_clusters.add(frozenset(grouped_files))
            print(f"Cluster: '{gram}' {len(grouped_files)}\t\t\t{', '.join(grouped_files)}")

