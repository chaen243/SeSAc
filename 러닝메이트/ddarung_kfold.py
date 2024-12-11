import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error
import time
from sklearn.ensemble import RandomForestRegressor

# 1. 데이터 로드
path = r"C:\Users\r2com\Desktop\수업자료\파일\daicon_bike\ddarung\\"
train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
submission_csv = pd.read_csv(path + "submission.csv")

train_csv = train_csv.fillna(train_csv.mean())  # 결측값 채우기
test_csv = test_csv.fillna(test_csv.mean())    # 결측값 채우기

# x, y 분리
x = train_csv.drop(['count'], axis=1)
y = train_csv['count']

# 2. K-Fold 설정
n_splits = 5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=123)
#shuffle이 False일 경우 순서대로 나눔


# 3. 모델 구성
model = RandomForestRegressor()


# 4. K-Fold를 사용한 교차 검증
start_time = time.time()
r2_scores = []

for train_index, test_index in kfold.split(x):
    x_train, x_test = x.iloc[train_index], x.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    # 모델 훈련
    model.fit(x_train, y_train)
    
    # 평가
    y_predict = model.predict(x_test)
    r2 = r2_score(y_test, y_predict)
    r2_scores.append(r2)

end_time = time.time()

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
rmse = RMSE(y_test, y_predict)


# R2 평균 및 표준편차 계산
mean_r2 = np.mean(r2_scores)
std_r2 = np.std(r2_scores)

print(f"Cross Validation R2 Scores: {r2_scores}")
print(f"Mean R2 Score: {mean_r2:.4f}")
print(f"Standard Deviation: {std_r2:.4f}")
print(f"RMSE : {rmse:.2f}")
print(f"걸린 시간: {round(end_time - start_time, 2)} 초")

# 5. 최종 모델 훈련 및 예측
model.fit(x, y)  # 전체 데이터로 최종 모델 훈련
y_submit = model.predict(test_csv)

# 6. 제출 파일 생성
submission_csv['count'] = y_submit

import time as tm
ltm = tm.localtime(tm.time())
save_time = f"{ltm.tm_year}{ltm.tm_mon}{ltm.tm_mday}{ltm.tm_hour}{ltm.tm_min}{ltm.tm_sec}" 
submission_csv.to_csv(path + f"submission_{save_time}{rmse:.3f}.csv", index=False)

print("done!")