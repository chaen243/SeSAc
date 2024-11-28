from bs4 import BeautifulSoup
import requests as req


url = 'https://search.naver.com/search.naver'
res = req.get(url, params={'query': '정처기'})
'''
if res.status_code == 200:
    soup = BeautifulSoup(res.text, 'html.parser')
    target_lst = []
    for element in soup.find_all(text = True): #모든텍스트를 가져옴
        if '정보처리기사' in element:
            target_lst.append(element.strip())

    print(target_lst)
else:
    print(f"요청실패 : {res.status_code}")
'''

#####try-except문   
import re 

url = 'https://search.naver.com/search.naver'

try:
    res = req.get(url, params={'query':'정처기'}, timeout=5) #5초 타임아웃 설정(5초 넘어가면 바로 except문으로/ try 성공하면 else로 감)
    res.raise_for_status() #응답 상태 코드가 200번대가 아닐 경우 예외 발생
except req.exceptions.RequestException as e:
    print(f"HTTP 요청 중 오류 발생 : {e}")
else:
    #성공적으로 데이터를 받았다면, HTML 파싱
    res.encoding = 'utf-8' #한글 인코딩 설정
    soup = BeautifulSoup(res.text, 'html.parser')

    #정규 표현식을 사용하여 '정보처리기사'를 포함한 텍스트 찾기
    target_lst = []
    pattern = re.compile(r'.*정보처리기사.*')     #'정보처리기사' 가 포함된 텍스트 찾기 위한 정규식 패턴

    #페이지에서 모든 텍스트 추출
    for element in soup.find_all(text= True): #모든 텍스트 요소 순회
        if pattern.match(element): #'정보처리기사'를 포함하는 텍스트만 필터링
            target_lst.append(element.strip()) #공백 제거 후 리스트에 추가
            # target_lst += element.strip() <- 문자를 하나씩 받음!
    #결과 출력
    if target_lst:
        print('정보처리기사 관련 텍스트 목록:')
        for item in target_lst:
            print(item)
    else:
        print('정보처리기사 관련 텍스트를 찾을 수 없습니다.')        



