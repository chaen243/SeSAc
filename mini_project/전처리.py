import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re


#데이터

path = 'C:\\Users\\r2com\\Desktop\\수업자료\\'

data = pd.read_csv(path + 'updated_file.csv')

# print(data)

print(data.columns)
#['place_name',  업체명
# 'category_name', 음식점 카테고리 분류
# 'category_group_name', 음식점 or 카페
# 'phone', 음식점 번호
# 'road_address_name', 도로명주소
# 'x', 경도
# 'y', 위도
# 'place_url', 카카오맵 url 
# 'distance_from_current'], 현위치(학원 기준) 거리


#결측치 채우기 (임의로 앞의 값으로  채움)
data = data.fillna(method= 'ffill')

#결측치 확인
# print(data.isna().sum())

#카테고리 갯수 확인



# 카테고리를 '>'로 나눈 뒤 DataFrame으로 변환
split_categories = data["category_name"].str.split(">", expand=True)

# 두 번째 카테고리부터 새로운 컬럼으로 저장
for i in range(0, split_categories.shape[1]):  # 2번째 인덱스(세 번째 카테고리)부터 반복
    data[f"category_{i-1}"] = split_categories[i].str.strip()  # 공백 제거 후 저장

# 결과 출력
# print(data)

#데이터 새로 저장
# data.to_csv("확인용1.csv", index=False)








