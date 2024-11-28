import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import re

#데이터

path = 'C:\\Users\\r2com\\Desktop\\수업자료\\mini_project\\'



user_ = pd.read_csv(path + '응답시트.csv')  #설문응답자료
file_ = pd.read_csv('C:\\Users\\r2com\\Desktop\\수업자료\\확인용1.csv') #식당자료

# print(user_)

print(file_.columns)
# Index(['id', 'place_name', 'category_name', 'category_group_code',
#        'category_group_name', 'phone', 'address_name', 'road_address_name',
#        'x', 'y', 'place_url', 'distance'],

# print(file_['category_group_name'].value_counts)

#비플페이 사용 안되는 사용처 술집 제거
new_file = file_[~file_['category_name'].str.contains("술집", na=False)]

# new_file.to_csv("newwww.csv", index=False)

cate = new_file['category_0'].unique()
print(cate)
# ['패스트푸드' '일식' '구내식당' '치킨' '패밀리레스토랑' '중식' '한식' '분식' '간식' '샤브샤브' '샐러드' '양식'
#  '아시아음식' '퓨전요리' '뷔페']



#user에 음식 카테고리 컬럼 생성 (가중치 부여를 위해)
for cate_name in cate:
    user_[cate_name] = 0
print(user_)

# weights = {1:5,2:3,3:1,4:5,5:3,6:1}

# # 카테고리별 가중치 합 계산
# result = {}

# for col_idx, weight in weights.items():
#     for category in user_.iloc[:, col_idx]:
#         result[category] = result.get(category, 0) + weight

# # 결과 정리
# result_df = pd.DataFrame(list(result.items()), columns=["카테고리", "가중치 합"])
# result_df = result_df.sort_values(by="가중치 합", ascending=False)

# # 결과 출력
# print(result_df)

df = pd.DataFrame(user_)

# 음식 카테고리별 컬럼 초기화 (중식, 양식, 한식, 일식, 디저트 등이 포함된다고 가정)
categories = ["중식", "양식", "한식", "일식", "디저트", "카페"]
for category in categories:
    df[category] = 0  # 초기값 0으로 설정

# 가중치 임의 설정
weights = {0:0,1:5,2:5,3:3,4:1,5:3,6:1}  # 첫번째, 두번째, 세번째 선호도 가중치

# 가중치 계산 및 반영
for idx, row in df.iterrows():
    for i, col in enumerate([1,2,3,4,5,6]):
        category = row[col]
        if category in categories:  # 해당 카테고리가 categories에 있을 때만
            df.at[idx, category] += weights[i]

# 결과 출력
print(df)

df.to_csv('확인2.csv', index=False)