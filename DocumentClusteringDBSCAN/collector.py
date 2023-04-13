#!/usr/bin/env python
import pandas as pd
import requests
from bs4 import BeautifulSoup
import time

article_list = [] # 기사가 담길 리스트 선언
for index in range(20):
    # page 를 변경하며 한페이지의 전체기사 목록을 가져오기
    url = 'https://news.daum.net/breakingnews/?page={}&regDate=20220707'.format(index)
    html = requests.get(url)
    soup = BeautifulSoup(html.content, "html.parser")
    news_section = soup.find('ul',{'class':'list_news2 list_allnews'})
    news_list = news_section.find_all('a',{'class':'link_txt'})
    if len(news_list) < 15: # page내 기사가 15개 미만이라면 마지막 page로 간주, 반복문 탈출!
        break

    for news in news_list:
        article_title = news.text
        article_url = news.get('href')
        try:
            article_html = requests.get(article_url)
            soup = BeautifulSoup(article_html.content, "html.parser")
            text_list = [text_tag.text for text_tag in soup.find_all('p',{'dmcf-ptype':"general"})]
            article_content = " ".join(text_list)
            # 기사 정보를 제목과 본문으로 저장하기 위한 dictionary 형태로 변환
            article = dict()
            article['title'] = article_title
            article['content'] = article_content
        except:
            continue

        article_list.append(article) # 기사 담기
    # 중간중간 잘 진행되는지 여부 확인을 위해 기사 개수가 10배수 일때마다 표시하기
    if len(article_list) % 10 == 0:
        print(index, len(article_list))
        time.sleep(0.5) # 가져오는 작업이 서버의 부담을 줄수있기에 잠깐씩 쉬어준다.

df = pd.DataFrame(article_list, columns=['title', 'content', 'nouns'])
df.to_pickle("daum_news.pkl")
print(df['title'])

