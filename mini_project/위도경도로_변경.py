import requests

# API 키 설정
API_KEY = "b3b3f651d175d25d1158c423890e4110"  # Kakao Developers에서 발급받은 REST API Key
headers = {"Authorization": f"KakaoAK {API_KEY}"}

# 주소 -> 위도/경도 변환 함수
def get_address(address):
    url = f"https://dapi.kakao.com/v2/local/search/address.json?query={address}"
    response = requests.get(url, headers=headers)
    
    if response.status_code == 200:
        result = response.json()
        documents = result.get("documents")
        if documents:
            # 첫 번째 검색 결과의 위도와 경도 반환
            latitude = documents[0]["y"]
            longitude = documents[0]["x"]
            return float(latitude), float(longitude)
        else:
            print("주소를 찾을 수 없습니다.")
            return None, None
    else:
        print(f"에러 발생: {response.status_code}")
        return None, None




# 테스트: 주소 입력
address = "서울 도봉구 마들로13길 61"
latitude, longitude = get_address(address)

if latitude and longitude:
    print(f"주소: {address}")
    print(f"위도: {latitude}, 경도: {longitude}")

    # 위도: 37.6532084846758, 경도: 127.04768378525 (창동역)
    # 위도: 37.6545399298261, 경도: 127.049926290286 (학원)