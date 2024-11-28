import requests
from bs4 import BeautifulSoup
'''
url = 'https://www.khan.co.kr/it/it-general/article/201707111831001'
res = requests.get(url)
print('응답코드:', res.status_code) #문제가 있는지 확인 None = 문제 없음
'''

'''
res = requests.get('http://google.com')
res.raise_for_status()

with open('mygoogle.html','w',encoding='utf-8') as f:
    f.write(res.text)
'''


'''
#크롤링한 데이터 확인
res = requests.get('http://openapi.seoul.go.kr:8088/6d4d776b466c656533356a4b4b5872/json/RealtimeCityAir/1/99')
resj = res.json()

#원하는 자료 확인
print(resj['RealtimeCityAir']['row'][0]['NO2'])



#서울 구 이름[MSRSTE_NM]과 해당구[IDEX_MVL]에 미세먼지 데이터만 가져와 출력
citys = resj['RealtimeCityAir']['row']


for city in citys:
    name = city['MSRSTE_NM']
    mise = city['IDEX_MVL']
    print(name,mise)

import requests as req
#
url = 'https://search.naver.com/search.naver'
res = req.get(url, params={'query' : 'SQL'})
#get -> response의 정보를 가져온것/ params(파라미터) -> 쿼리 ~~~/정처기 로 검색해준것

if res.status_code == 200:
    #정상적인 반응 (code가 200일때)
    soup = BeautifulSoup(res.text, 'html.parser')
    #response로 받아온 정보들은 html.parser로 짤라서 정렬
    #정보처리기사 포함된 텍스트를 모두 찾기
    target_lst = []
    #빈공간부터 만듬
    for element in soup.find_all(text=True): #모든 텍스트를 가져옴.
        #beautifulsoup으로 자른 것을 모두 찾아주는(find_all)것.(text이면 모두 다)
        #변수 element로 받아옴
        if 'SQL' in element:
            #만약 변수 element안에 '정보처리기사'라는 단어가 있다면,
            target_lst.append(element.strip())
            #target_lst에 쌓아줘. strip = 문자열의 시작과 끝에서 공백을 제거한 후 반환.
    print(target_lst)
else:
    print(f'요청실패 : {res.status_code}')            
'''
import os
import sys
import urllib.request
client_id = "RzKVFU8prLxgz7d3lPh1"
client_secret = "8bBQOSp0FF"
encText = urllib.parse.quote("경복궁")
url = "https://openapi.naver.com/v1/search/blog?query=" + encText # JSON 결과
# url = "https://openapi.naver.com/v1/search/blog.xml?query=" + encText # XML 결과
request = urllib.request.Request(url)
request.add_header("X-Naver-Client-Id",client_id)
request.add_header("X-Naver-Client-Secret",client_secret)
response = urllib.request.urlopen(request)
rescode = response.getcode()
if(rescode==200):
    response_body = response.read()
    print(response_body.decode('utf-8'))
else:
    print("Error Code:" + rescode)