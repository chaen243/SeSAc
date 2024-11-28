import requests
res = requests.get('http://naver.com')
print('응답코드 :', res.status_code) #응답코드 : 200 200대면 정상 반응

res2 = requests.get('http://naver.com/user') #로그인을 해야만 나오는 사이트
print('응답코드 :', res2.status_code) #응답코드 : 404

# - 응답 상태코드 구분
#     - 1xx : Informational (단순 정보 제공)
#     - 2xx : Successful (성공)
#     - 3xx : Redirect (추가 정보 필요)
#     - 4xx : Client error
#     - 5xx : Server error (주로 서버점검등의 상황)

#403 에러는 접속 차단.
# 요청시 헤더정보를 크롬으로 지정(될때도 있고 안될때도 있음.)
request_headers = {
'User-Agent' : ('Mozilla/5.0 (Windows NT 10.0;Win64; x64)\
AppleWebKit/537.36 (KHTML, like Gecko) Chrome/71.0.3578.98\
Safari/537.36'), }

url = "https://search.naver.com/search.naver"
response = requests.get(url,headers = request_headers)

print(response)
###########403#############

