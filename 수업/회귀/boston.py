import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split, cross_val_score
                                    #^train과 test데이터 분리
from sklearn.metrics import mean_squared_error
from sklearn import datasets



#데이터
X, y = datasets.fetch_openml('boston', return_X_y=True)

# print(X.columns)
#Index(['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX',
    #    'PTRATIO', 'B', 'LSTAT'],
    #   dtype='object')

# print(X.describe()) #결측치 확인 (없음) ,대략적인 데이터의 분포 확인 가능

#              CRIM          ZN       INDUS         NOX          RM         AGE         DIS         TAX     PTRATIO           B       LSTAT
# count  506.000000  506.000000  506.000000  506.000000  506.000000  506.000000  506.000000  506.000000  506.000000  506.000000  506.000000
# mean     3.613524   11.363636   11.136779    0.554695    6.284634   68.574901    3.795043  408.237154   18.455534  356.674032   12.653063
# std      8.601545   23.322453    6.860353    0.115878    0.702617   28.148861    2.105710  168.537116    2.164946   91.294864    7.141062
# min      0.006320    0.000000    0.460000    0.385000    3.561000    2.900000    1.129600  187.000000   12.600000    0.320000    1.730000
# 25%      0.082045    0.000000    5.190000    0.449000    5.885500   45.025000    2.100175  279.000000   17.400000  375.377500    6.950000
# 50%      0.256510    0.000000    9.690000    0.538000    6.208500   77.500000    3.207450  330.000000   19.050000  391.440000   11.360000
# 75%      3.677083   12.500000   18.100000    0.624000    6.623500   94.075000    5.188425  666.000000   20.200000  396.225000   16.955000
# max     88.976200  100.000000   27.740000    0.871000    8.780000  100.000000   12.126500  711.000000   22.000000  396.900000   37.970000

# print(X.info()) #dtypes: category(2), float64(11)

#결측치 확인
# print(X.isnull().sum()) #없음
# print(X.isna().sum()) #없음  둘다 결과는 동일하게 출력.

#y에 대한 히스토그램 그리기
import seaborn as sns #통계학 분석 패키지

# sns.set_theme(rc={'figure.figsize':(15,10)})
# plt.hist(y, bins=30) #bins = 막대 갯수
# plt.xlabel('boston')
# plt.show()

# corrleation_matrix = X.corr().round(2)
# sns.heatmap(data = corrleation_matrix,annot=True) #true 소숫점 표기 False= 숫자 표출 안됨.
#                                                    #밝을수록 연관성이 있음. 1은 자기자신., -1과 1에 가까울수록 연관성 up
#                                                    #상관관계에 대한 히트맵. 
# plt.show()


#집값과 관련이 있는지 확인하는 그림
# plt.figure(figsize=(10,5))

# features = ['LSTAT','RM','DIS']

# for i,col in enumerate(features):
#     plt.subplot(1,len(features),i+1) #subplot은 확장이 가능 subplots는 확장 안됨
#     x = X[col]
#     plt.scatter(x,y,marker='o',color='#e35f62') #hex color
#     plt.title('Variation in House Price')
#     plt.xlabel(col)
#     plt.ylabel('House Price in $1000')
# plt.show()    

#컬럼확장
# column_sels = 
# x = X.RM #(506,) ->행 506개,가변공간
#sklearn에서는 꼭 2D 형태여야 동작가능 (n by1 or 1 by n)형태여야함
# 그래서 모델을 돌릴때는 항상 2d 형태로 바꿔줘야함
# x = np.array(x).reshape(-1,1)
# y = np.array(y).reshape(-1,1)

# print(x.shape) #(506, 1)
print(y.shape) #(506, 1) 

#모의고사, 수능 데이터로 쪼개기
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=5) #random_state == random_seed
#(X_train,y_train)/(X_test,y_test)가 서로 짝. 꼭 순서 지켜서 써야함
#행렬을 쓸때는 주로 x를 대문자로 사용 (암묵적인 약속)
# print(X_train.shape) #(404, 1)
# print(X_test.shape) #(102, 1)
# print(y_train.shape) #(404, 1)
# print(y_test.shape) #(102, 1)

#훈련
reg = LinearRegression()
reg.fit(X_train,y_train)
y_test_predict = reg.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test,y_test_predict))
r2 = round(reg.score(X_test,y_test),2)

#결과
print(r2) #0.43(train data로 했을때)   0.69 (test 데이터로 했을때)
#train 데이터로 predict를 하면 과적합이 올 수 있음.

prediction_space = np.linspace(min(X_train), max(X_train)).reshape(-1,1)
plt.scatter(X_train,y_train)
plt.plot(prediction_space, reg.predict(prediction_space),color='black',linewidth=3 )
plt.ylabel('value of house/1000($)')
plt.xlabel('number of rooms')
plt.show()