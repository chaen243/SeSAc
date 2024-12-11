#https://dacon.io/competitions/open/235576/mysubmission

#사용 라이브러리 import
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split,KFold,cross_val_score
from sklearn.metrics import r2_score, mean_squared_error
import time
from sklearn.linear_model import LinearRegression, Perceptron
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler, RobustScaler


#1. 데이터
path = r"C:\Users\r2com\Desktop\수업자료\파일\daicon_bike\ddarung\\"
#문자열 앞에 r을 사용하면 문자열안의 이스케이프문자들을 인식하지 않고 문자열 그대로 받음


# pandas에서 1차원- Series, 2차원이상은 DataFrame이라고 함.

#데이터 불러오기
train_csv = pd.read_csv(path + "train.csv", index_col=0) # \ \\ / // 다 가능( 예약어 사용할때 두개씩 사용) 인덱스컬럼은 0번째 컬럼이다라는뜻.
#print(train_csv)
test_csv = pd.read_csv(path +"test.csv", index_col=0)
#print(test_csv)
submission_csv = pd.read_csv(path + "submission.csv") 
#print(submission_csv)


# print(train_csv.isnull().sum()) 결측치 확인
train_csv = train_csv.fillna(train_csv.mean())  #중간값으로 채움
test_csv = test_csv.fillna(test_csv.mean())   #중간값으로 채움
# print(train_csv.isnull().sum()) 결측치가 사라져야함
#hour                      0
# hour_bef_temperature      0
# hour_bef_precipitation    0
# hour_bef_windspeed        0
# hour_bef_humidity         0
# hour_bef_visibility       0
# hour_bef_ozone            0
# hour_bef_pm10             0
# hour_bef_pm2.5            0
# count                     0
#print(train_csv.info())
#print(train_csv.shape)      #(1328, 10)
#print(test_csv.info()) # 717 non-null


################# x와 y를 분리 ###########
x = train_csv.drop(['count',], axis=1)
#print(x)
y = train_csv['count'] #맞춰야할 값
#print(y)

#훈련을 위해 train,test 데이터 분리
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size= 0.72,  shuffle= True, random_state= 6) 

# print(train_csv.index)



#2. 모델구성
model =  RandomForestRegressor()

#3. 훈련

start_time = time.time()
model.fit(x_train, y_train)
end_time = time.time()




#4.평가, 예측
loss = model.score(x_test, y_test) #model.score에서는 모델에 대한 r2값 자동 반환 (모델마다 다름)
y_predict = model.predict(x_test) #x_test값을 입력받아 예측한 값 저장
r2 = r2_score(y_test, y_predict) #y_test,y_predict를 예측한 값 저장


#model.score(x_test, y_test):
# **x_test와 y_test**를 입력으로 받습니다.
# model.score 내부적으로 다음을 수행합니다:
# x_test를 이용해 예측값 y_predict를 생성합니다:
# y_predict=model.predict(x_test)
# 생성된 y_predict와 실제값 y_test를 이용해 R² 값을 계산합니다:


# r2_score(y_test, y_predict):
# **y_test(실제값)과 y_predict(모델이 생성한 예측값)**를 입력으로 받습니다.
# 이미 만들어진 y_predict를 이용하여 동일한 R² 값을 계산합니다.

#결론적으로는 loss와 r2값은 같은 값을 반환

y_submit = model.predict(test_csv) #test_csv에 대한 예측값
       

#print(submission_csv.shape)
print("model.score :", loss)
print("R2 스코어 :", r2)
print("걸린 시간 :", round(end_time - start_time, 2), "초")

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
rmse = RMSE(y_test, y_predict)
print("RMSE : ", rmse)

y_submit = (y_submit.round(0).astype(int)) #실수를 반올림한 정수로 나타내줌.


####### submission.csv 만들기 (count컬럼에 값만 넣어주면 됨) #####
submission_csv['count'] = y_submit
print(submission_csv)


import time as tm
ltm = tm.localtime(tm.time())
save_time = f"{ltm.tm_year}{ltm.tm_mon}{ltm.tm_mday}{ltm.tm_hour}{ltm.tm_min}{ltm.tm_sec}" 
submission_csv.to_csv(path + f"submission_{save_time}{rmse:.3f}.csv", index=False)

print('done!')



