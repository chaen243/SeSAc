from sklearn.model_selection import KFold
import numpy as np

#예시 데이터
X_data = np.arange(10).reshape(10,1)
y_target = np.arange(10)

#K-fold 객체 생성(5개의 객체로 나눔.)
kf = KFold(n_splits=5, shuffle=True,random_state=42)

#fold 정보 확인.
fold_idx = 1
for train_index, test_index in kf.split(X_data):
    print(f'Fold {fold_idx} : ')
    print(f'Train indices : {train_index}')
    print(f'Test imdices : {test_index}')
    print(f'Train data : {X_data[train_index].flatten()}')
    print(f'Test data : {X_data[test_index].flatten()}')
    fold_idx += 1
