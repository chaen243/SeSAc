from geopy.geocoders import Nominatim

#예제 데이터 : df_shake
#컬럼 정보 : name, branch, addr

# 위도, 경도 반환하는 함수
def geocoding(address):
    try:
        geo_local = Nominatim(user_agent='South Korea')  #지역설정
        location = geo_local.geocode(address)
        geo = [location.latitude, location.longitude]
        return geo

    except:
        return [0,0]


# 실행
for idx,addr in enumerate(tqdm(df_shake.addr)):
    df_shake.loc[idx,'latitude'] = geocoding(addr)[0]
    df_shake.loc[idx,'longitude'] = geocoding(addr)[1]