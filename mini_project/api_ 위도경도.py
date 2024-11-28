# %%

import pandas as pd
import json
import requests
import warnings
warnings.filterwarnings('ignore')

changdong = '서울 도봉구 마들로11길 77'
#changdong_ll = 위도: 37.6532084846758, 경도: 127.04768378525

restaurants2 = pd.read_csv('C:\\Users\\r2com\\Desktop\\수업자료\\mini_project\\restaurants3.csv')

headers = {"Authorization": "KakaoAK b3b3f651d175d25d1158c423890e4110"}

geocoding_url = 'https://dapi.kakao.com/v2/local/search/address.json?query='
# key값 KaKaoAK + rest API 값 넣기

test = restaurants2.loc[0, 'road_address_name']


# print(result)

# print(latitude)

coordinates = []

for idx, address in enumerate(restaurants2['road_address_name']):
    url = geocoding_url + address
    result = json.loads(requests.get(url, headers=headers).text)
    
    if result['documents']:  # 결과가 있을 경우

        match_first = result['documents'][0]['address']
        latitude = float(match_first['y'])
        longitude = float(match_first['x'])
        coordinates.append({'latitude': latitude, 'longitude': longitude}) #latitude 위도
    else:
        coordinates.append({'latitude': None, 'longitude': None})  # 주소가 없는 경우 처리

######## DataFrame에 좌표 추가 ###########
coords_df = pd.DataFrame(coordinates)
Library = pd.concat([restaurants2, coords_df], axis=1)

print(Library.head())
# print(restaurants2.isna().sum())









# ######## 위도 경도로 거리 계산 #########
# from haversine import haversine


# # 위경도 입력
# changdong = (37.6532084846758, 127.04768378525)  #Latitude, Longitude

# c_distance
# # 위도: 37.6532084846758, 경도: 127.04768378525
# # 거리 계산
# print(haversine(Seoul, Toronto, unit = 'km'))


#https://www.google.com/maps/dir/%EC%84%9C%EC%9A%B8%ED%8A%B9%EB%B3%84%EC%8B%9C+%EC%B0%BD%EB%8F%99+135%EB%B2%88%EC%A7%80+%EC%B0%BD%EB%8F%99//data=!3m1!4b1!4m9!4m8!1m5!1m1!1s0x357cb94e50956a99:0x311655df1d27c6c6!2m2!1d127.047945!2d37.65276!1m0!3e3?entry=ttu&g_ep=EgoyMDI0MTExOS4yIKXMDSoASAFQAw%3D%3D










# ### 지도에 위치 표시해서 저장 & 표출 & url로 연결###
import folium
from IPython.display import display 


center = [37.6532084846758, 127.04768378525] #현위치 경도 (창동역)
map = folium.Map(location= center, zoom_start= 16)

for i in restaurants2.index[:len(restaurants2)]:      # 데이터 개수만큼 반복한다.
    html = f'<a href="{restaurants2.loc[i, "place_url"]}" target="_blank">{restaurants2.loc[i, "place_name"]}</a>'
    #popup에 링크 추가
    popup = folium.Popup(html, max_width=300)
    folium.Circle(
        location = restaurants2.loc[i, ['y', 'x']],
        tooltip = restaurants2.loc[i, 'place_name'], #마우스 올릴때 이름 표시
        popup= popup,
        radius = 8,
        color = 'blue',
        fill_color = 'blue'
      ).add_to(map)

display(map)

# map.save("map.html")




# %%
