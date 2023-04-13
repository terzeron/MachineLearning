#!/usr/bin/env python

import konlpy

from konlpy.tag import Okt
okt = Okt()

text = "한글 자연어 처리는 재밌다 이제부터 열심히 해야지ㅎㅎㅎ"
print("형태소분석:", okt.morphs(text)) # 형태소 분석
print("형태소분석(어간추출):", okt.morphs(text, stem=True)) # 형태소 분석 후 어간 추출

print("명사:", okt.nouns(text))
print("어절:", okt.phrases(text))

print("품사태깅:", okt.pos(text))
print("품사태깅(결합):", okt.pos(text, join=True))


from konlpy.corpus import kolaw
from konlpy.corpus import kobill

print(kolaw.open('constitution.txt').read()[:50])
print(kobill.open('1809890.txt').read()[:50])
