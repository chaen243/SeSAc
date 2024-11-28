import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split,KFold,cross_val_score, StratifiedKFold, cross_val_predict
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from xgboost import XGBClassifier



s_time = time.time()

#데이터
path = 'C:\\Users\\r2com\\Downloads\\archive\\'
df = pd.read_csv(path + 'diabetes.csv')
# print(df)

X = df.drop(['Outcome'], axis=1)
# print(X)
y = df['Outcome']
# print(y)
# print(y.value_counts())
#이진분류 문제로 y의 갯수 확인
#Outcome 
# 0    500
# 1    268

# print(X.shape) #(768, 8)
# print(y.shape) #(768,)

########데이터 정보 확인#########
# * Pregnancies : 임신 횟수
# * Glucose : 2시간 동안의 경구 포도당 내성 검사에서 혈장 포도당 농도
# * BloodPressure : 이완기 혈압 (mm Hg)
# * SkinThickness : 삼두근 피부 주름 두께 (mm), 체지방을 추정하는데 사용되는 값
# * Insulin : 2시간 혈청 인슐린 (mu U / ml)
# * BMI : 체질량 지수 (체중kg / 키(m)^2)
# * DiabetesPedigreeFunction : 당뇨병 혈통 기능
# * Age : 나이
# * Outcome : 당뇨병 여부, 768개 중에 268개의 결과 클래스 변수(0 또는 1)는 1이고 나머지는 0입니다.
#   - 0 : 당뇨병 없음
#   - 1 : 당뇨병 있음

# print(df.info()) #결측치없는것 확인.

#  #   Column                    Non-Null Count  Dtype
# ---  ------                    --------------  -----
#  0   Pregnancies               768 non-null    int64
#  1   Glucose                   768 non-null    int64
#  2   BloodPressure             768 non-null    int64
#  3   SkinThickness             768 non-null    int64
#  4   Insulin                   768 non-null    int64
#  5   BMI                       768 non-null    float64
#  6   DiabetesPedigreeFunction  768 non-null    float64
#  7   Age                       768 non-null    int64
#  8   Outcome                   768 non-null    int64
# dtypes: float64(2), int64(7)

# print(df.describe())
#        Pregnancies     Glucose  BloodPressure  SkinThickness     Insulin         BMI  DiabetesPedigreeFunction         Age     Outcome
# count   768.000000  768.000000     768.000000     768.000000  768.000000  768.000000                768.000000  768.000000  768.000000
# mean      3.845052  120.894531      69.105469      20.536458   79.799479   31.992578                  0.471876   33.240885    0.348958
# std       3.369578   31.972618      19.355807      15.952218  115.244002    7.884160                  0.331329   11.760232    0.476951
# min       0.000000    0.000000       0.000000       0.000000    0.000000    0.000000                  0.078000   21.000000    0.000000
# 25%       1.000000   99.000000      62.000000       0.000000    0.000000   27.300000                  0.243750   24.000000    0.000000
# 50%       3.000000  117.000000      72.000000      23.000000   30.500000   32.000000                  0.372500   29.000000    0.000000
# 75%       6.000000  140.250000      80.000000      32.000000  127.250000   36.600000                  0.626250   41.000000    1.000000
# max      17.000000  199.000000     122.000000      99.000000  846.000000   67.100000                  2.420000   81.000000    1.000000


##########이상치 처리#############
#Pregnancies에서 max값이 17회 인것을 확인 (이상치..?)
# 'SkinThickness', 'BloodPressure','Insulin'에서 0은 나올수 없는 치수로 판단. 수치 처리 필요하다고 판단.

# print(X['Pregnancies'].value_counts())
#10회 이상은 이상치로 판단. 10회 이상은 평균치로 이상치 처리


mean_s = int(X['SkinThickness'][X['SkinThickness']].mean()) #skinThickness 값이 0인 컬럼들은 평균값으로 대체.
X['SkinThickness'] = np.where(X['SkinThickness'] == 0, mean_s, X['SkinThickness']).astype(int)
# print((X['SkinThickness']).value_counts()) #0 없어진것 확인

mean_b = int(X['BloodPressure'][X['BloodPressure']!= 0].mean()) #'BloodPressure' 값이 0인 컬럼들은 평균값으로 대체.
X['BloodPressure'] = np.where(X['BloodPressure'] == 0, mean_s, X['BloodPressure']).astype(int)
# print((X['BloodPressure']).value_counts()) #0 없어진것 확인


mean_p = int(X['Pregnancies'][X['Pregnancies'] <= 10].mean()) #10회 이하의 값들만으로 평균치를 구함.
X['Pregnancies'] = np.where(X['Pregnancies'] > 10, mean_p, X['Pregnancies']).astype(int)
# print((X['Pregnancies']).value_counts()) #10회이하의 정수형으로 변경 된 것 확인.


# print(type(X['Insulin']))
# mean_i = int(X['Insulin'][X['Insulin'] != 0].mean()) #Insulin 값이 0인 컬럼들은 평균값으로 대체.
# X['Insulin'] = np.where(X['Insulin'] == 0, mean_i, X['Insulin']).astype(int)





#######나머지 이상치 처리#######
def fit_outlier(X):  
    X = pd.DataFrame(X)
    for label in X:
        series = X[label]
        q1 = series.quantile(0.25)      
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        upper_bound = q3 + iqr
        lower_bound = q1 - iqr
        
        series[series > upper_bound] = np.nan
        series[series < lower_bound] = np.nan
        print(series.isna().sum())
        series = series.interpolate()
        X[label] = series
        
    X = X.fillna(X.mean())
    return X
# X = fit_outlier(X)
# print(X)
##########이상치 처리#############

#데이터 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42, shuffle=True, stratify= y) #67

n_splits=5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=123)

parameters = {
    'n_estimators' : [100],
    'learning_rate' : [0.1],
    'max_depth' : [3]
}


# model = KNeighborsClassifier(n_neighbors= 9, weights='distance', algorithm= 'auto')
# model = RidgeClassifier(alpha=1.0, tol= 0.01, max_iter= 20,random_state= 123)
# model = XGBClassifier(**parameters)
model = RandomForestClassifier(n_estimators= 3000, max_depth=10, criterion= 'log_loss')






#3. 훈련
model.fit(X_train, y_train )

#4. 평가, 예측
scores = cross_val_score(model, X, y, cv=kfold)
y_predict = cross_val_predict(model, X_test, y_test, cv= kfold)
acc= accuracy_score(y_test, y_predict)
print('cross_val_predict ACC :', acc)
print("acc :", scores, "\n 평균 acc :", round(np.mean(scores),4))




#print(y_predict)
e_time = time.time()

#모델링 시간 표출
print('걸린시간 : ',round(np.mean(e_time - s_time),2),'초')
