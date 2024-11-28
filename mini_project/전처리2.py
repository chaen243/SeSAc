import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re


path = 'C:\\Users\\r2com\\Desktop\\수업자료\\'

data = pd.read_csv(path + '확인용1.csv')

# print(data.columns)
# Index(['place_name', 'category_name', 'category_group_name', 'phone',
#        'road_address_name', 'x', 'y', 'place_url', 'distance_from_current',
#        'category_-1', 'category_0', 'category_1', 'category_2', 'category_3'],
#         dtype='object')


print(data['category_0'].value_counts())

print(data['category_1'].value_counts())


#한,양,중,일,간식분식,etc...?
# 한식         50
# 술집         15 cut
# 간식          9
 # 일식          8
# 분식          8
# 치킨          7
# 중식          3
# 패스트푸드       2
# 샐러드         2
# 양식          2
# 뷔페          2
# 구내식당        1  cut
# 패밀리레스토랑     1 
# 샤브샤브        1
# 아시아음식       1
# 퓨전요리        1

print(data['category_0'].count()) #113.

###one hot encoding (고유값의 갯수가 적을때, 순서나 크기가 없을때)
###label encoding (고유값의 갯수가 많을때, 순서나 크기가 의미가 있을때 사용) (위생등급 데이터)

from sklearn.preprocessing import OneHotEncoder

category_ = pd.get_dummies(data["category_0"])

# 결과 출력
print(category_)

########위생등급 데이터 label encoding 하기##########



#############추천할 음식점 컬럼 만들기#########
data['1위 식당'] = np.nan
data['2위 식당'] = np.nan
data['3위 식당'] = np.nan
data['4위 식당'] = np.nan
data['5위 식당'] = np.nan

print(data)