# import matplotlib.pyplot as plt
# '''
# # 장르 카운트 결과
# genre_counts = {'로맨스': 0, '사랑': 1, '연애': 0, '판타지': 0, '마법': 0, '모험': 0, '액션': 0, '전투': 0, '싸움': 0,
#                 '전쟁': 0, '드라마': 0, '감동': 0, '미스터리': 0, '추리': 0, '스릴러': 0, '긴장': 0, '코미디': 0, '유머': 0,
#                 '공포': 0, '귀신': 0, 'SF': 0, '미래': 0, '우주': 0, '로봇': 0, '과학': 0, '기술': 0, '계약': 2, '복수': 1, '결혼': 0}


# #데이터 준비
# genre = list(genre_counts.keys())
# counts = list(genre_counts.values())

# plt.rcParams['font.family'] ='Malgun Gothic'
# plt.rcParams['axes.unicode_minus'] =False

# #막대그래프 생성
# plt.figure(figsize = (10, 6))
# plt.barh(genre, counts, color= 'skyblue')
# #제목과 라벨 설정
# plt.title('장르별 웹툰 제목 카운트')
# plt.xlabel('카운트')
# plt.ylabel('장르')

# #그래프 출력
# plt.show()
# '''
# '''
# # from wordcloud import WordCloud
# import matplotlib.pyplot as plt
# import re

# # 제공된 웹툰 제목들
# titles = ['남편을 버렸다', '빌런의 순정', '후회조차 사치인 당신들에게', '못난이 아내',
#           '프레스턴 경의 비밀 가정교사', '교환 아내', '계약혼', '사랑해서 미칠 것 같아',
#           '남편의 정부가 어느 날 황후가 되었다', '마교전선 비룡십삼대', '49번 남았습니다, 스승님!',
#           '내 남편을 뺏은 동생에게 복수하는 방법', '어차피 시한부, 후회는 없습니다',
#           '전무님과 아이를 키웁니다', '계약이 끝나는 날']

# # 제목에서 단어 추출
# all_words = []


# for title in titles:
#     clean_title = re.sub(r'[^\w\s]', '', title)  # 특수문자 제거
#     words = clean_title.split()  # 단어로 분리
#     # all_words.extend(words)  # 모든 단어를 리스트에 추가 

#     # all_words.append(words)
#     all_words+=words

# print(all_words)

# from wordcloud import WordCloud
# import matplotlib.pyplot as plt
# import re

# # 제공된 웹툰 제목들
# titles = ['남편을 버렸다', '빌런의 순정', '후회조차 사치인 당신들에게', '못난이 아내',
#           '프레스턴 경의 비밀 가정교사', '교환 아내', '계약혼', '사랑해서 미칠 것 같아',
#           '남편의 정부가 어느 날 황후가 되었다', '마교전선 비룡십삼대', '49번 남았습니다, 스승님!',
#           '내 남편을 뺏은 동생에게 복수하는 방법', '어차피 시한부, 후회는 없습니다',
#           '전무님과 아이를 키웁니다', '계약이 끝나는 날']

# # 제목에서 단어 추출
# all_words = []

# for title in titles:
#     clean_title = re.sub(r'[^\w\s]', '', title)  # 특수문자 제거
#     words = clean_title.split()  # 단어로 분리
#     all_words.extend(words)  # 모든 단어를 리스트에 추가 
#     #append = 하나의

# # 단어를 하나의 문자열로 결합
# text = ' '.join(all_words)

# # 단어 클라우드 생성
# wordcloud = WordCloud(font_path='/usr/share/fonts/truetype/nanum/NanumGothic.ttf', width=800, height=400, background_color='white').generate(text)

# # 단어 클라우드 시각화
# plt.figure(figsize=(10, 5))
# plt.imshow(wordcloud, interpolation='bilinear')
# plt.axis('off')  # 축 제거
# plt.show()
# '''

# import requests
# from bs4 import BeautifulSoup
# import pandas as pd
# from collections import Counter
# import matplotlib.pyplot as plt

# # URL 정의
# url = 'https://quotes.toscrape.com'

# # 웹페이지 요청
# response = requests.get(url)
# print(response) #<Response [200]>

# #HTML 파싱
# soup = BeautifulSoup(response.text, 'html.parser')

# #명언과 저자, 태그 정보 추출
# quotes = soup.select('div.quote')
# #qoutes = soup.find_all('div', class_ = 'quote') #둘다 똑같!

# #데이터를 저장할 리스트
# text_list = []
# tags_list = []
# quotes_list = []
# #명언과 저자 정보 수집
# for i in quotes:
#     text = i.select_one('span.text').text
#     author = i.select_one('span small').text
#     tags = [tag.text for tag in i.select('.tag')] #tag들을 다 불러옴
#     text_list.append((text,author))
#     tags_list.append(tags)

# #데-이터 프레임 생성
# df = pd.DataFrame(quotes_list, columns=['명언','저자'])

# #태그 정보를 dataFrame에 추가
# df['tags']= tags_list
# #결과 출력
# print(df.head())




# # 태그 빈도수 분석
# all_tags = [tag for sublist in tags_list for tag in sublist]
# tag_count = Counter(all_tags)

# # 상위 10개 태그 출력
# top_tags = tag_count.most_common(10)
# print("\nTop 10 Tags:")
# for tag, count in top_tags:
#     print(f"{tag}: {count}")

# # 저자별 명언 빈도수 계산
# author_count = Counter(df['저자'])

# # 상위 10명 저자별 명언 빈도수 시각화
# top_authors = author_count.most_common(10)
# author_names, author_freq = zip(*top_authors)
# # author_names = 

# plt.figure(figsize=(12, 6))
# plt.bar(author_names, author_freq, color='skyblue')
# plt.title('Top 10 Authors by Quote Frequency')
# plt.xlabel('Author')
# plt.ylabel('Frequency')
# plt.xticks(rotation=45, ha='right')
# plt.show()


import requests
import pandas as pd
from bs4 import BeautifulSoup

book_list = []
#순위 제목 저자 

# 페이지 수를 반복하면서 데이터를 가져옵니다
for page_num in range(1, 11):  # 예시로 1~10페이지까지 크롤링
    url = f'https://www.yes24.com/main/default.aspx?page={page_num}'  # 페이지 번호를 URL에 추가
    res = requests.get(url)
    soup = BeautifulSoup(res.text, 'html.parser')

    books = soup.select('li.tp02')

    for book in books:
        rank = book.select('strong')[0].text.strip()
        name = book.select('strong')[1].text.strip()
        author = book.select('em')[1].text.strip()
        book_url = 'https://www.yes24.com' + book.select_one('a')['href']  # 책 상세 페이지 URL

        # 책 정보 리스트에 추가
        book_list.append([rank, name, author, book_url])

# 데이터프레임으로 변환
df = pd.DataFrame(book_list, columns=['순위', '제목', '저자', '링크'])

# Excel로 저장
df.to_excel('./yes24_best_books_all_pages.xlsx', index=False)

# 확인
print(df.head())


