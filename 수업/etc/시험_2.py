# <웹크롤링숙제: 총 3문제>
# 동적 크롤링 2문제 + 각자 원하는 사이트 1개
# 강사님 이메일: ryp1662@gmail.com
# 기한: 11월 21일(목요일) 24시까지

#1번
# 문제: Books to Scrape 사이트에서 책 제목과 가격 추출
# 목표: https://books.toscrape.com/ 웹사이트에서 상위 10개의 책 제목과 가격을 추출합니다.
# 각 책의 제목과 가격을 출력합니다.
# 요구 사항:
# 웹사이트: https://books.toscrape.com/
# 목표: 웹사이트에서 제공하는 책들의 제목과 가격을 추출하여 출력합니다.
# 제출 내용:
# o    Python 코드
# o    각 책의 제목과 가격을 출력한 결과


'''
import requests
import pandas as pd
from bs4 import BeautifulSoup

url = 'https://books.toscrape.com/'

res = requests.get(url)
# print('응답코드:', res.status_code) #응답코드: 200 문제 없음
res.encoding = 'utf-8'

soup = BeautifulSoup(res.text, 'html.parser')
book_list = []

books = soup.select('article.product_pod')

# print(books) 
for book in books:
    title = book.select_one('h3 a')['title']
    price = book.select_one('p.price_color')
 
    book_list.append([title,price])



df = pd.DataFrame(book_list, columns= ['제목','가격'])

print(df)



'''



# #2번
# 문제: 네이버 뉴스에서 "Python" 관련 최신 기사 제목과 링크 크롤링
# 목표:
# 네이버 뉴스에서 "Python" 관련 최신 기사를 크롤링하여, 각 기사의 제목과 링크를 추출합니다.
# 상위 10개의 기사를 추출하고, 제목과 링크를 출력합니다.
# 요구 사항:
# 웹사이트: https://search.naver.com/search.naver?&where=news&query=python
# 목표:
# 네이버 뉴스 검색 결과에서 "Python" 관련 기사를 검색하고, 상위 10개의 기사의 제목과 링크를 추출합니다.
# 제출 내용:
# o    Python 코드
# o    상위 10개의 기사의 제목과 링크 출력 결과

import requests
import pandas as pd
from bs4 import BeautifulSoup

url = 'https://search.naver.com/search.naver?&where=news&query=python'
res = requests.get(url)
print('응답코드:', res.status_code) #응답코드: 200 문제 없음
# res.encoding = 'utf-8'

# soup = BeautifulSoup(res.text, 'html.parser') 
# news_list = []
# news = soup.select('li.bx')
# # print(news)



# for new in news:
#       # news_title = new.select_one('a.news_tit')['title']
#       news_title = new.find('news_contents').find('a.title')
#     #   link = 'https://search.naver.com/search.naver?&where=news&query=python' + new.select_one('a')['href']
#       news_list.append([news_title])



# df = pd.DataFrame(news_list, columns= ['제목'])

# print(df)