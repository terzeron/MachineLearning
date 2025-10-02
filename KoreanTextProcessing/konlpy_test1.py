#!/usr/bin/env python

# https://konlpy.org/en/latest/

from konlpy.tag import Kkma
from konlpy.utils import pprint
kkma = Kkma()

# 구둣점으로 문장 분할함
pprint(kkma.sentences(u'네, 안녕하세요. 반갑습니다.'))
pprint(kkma.sentences(u'네 안녕하세요 반갑습니다'))

# 단어만 분리
pprint(kkma.nouns(u'질문이나 건의사항은 깃헙 이슈 트래커에 남겨주세요.'))

# 형태소 분석
pprint(kkma.pos(u'오류보고는 실행환경, 에러메세지와함께 설명을 최대한상세히!^^'))