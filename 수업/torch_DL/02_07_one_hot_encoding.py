import torch
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd

df = pd.read_csv(r'C:\Users\r2com\Desktop\수업자료\파일\one_hot_test.csv', names = ["length", "width", "height", "volume", "class_name"])
print(df.head())

dataset = df.drop(columns=['class_name']).values

y = df['class_name'].values
print(y)

X = dataset.copy()
print(X)

# case 1) Sklearn
# Label Encoder는 독립 변수가 아닌 종속 변수(라벨)에 대해 사용한다. 문자열이나 정수로된 라벨 값을  0  ~  K−1 까지의 정수로 변환.
e = LabelEncoder()
e.fit(y) # 텍스트 -> 숫자
print("Label Class String : {}".format(y))
Y = e.transform(y)
print("Label Class Int: {}".format(Y))


## case 2) pandas
one_hot_label=pd.get_dummies(y)
# print("case 2 one_hot_label : ", one_hot_label)
# print(one_hot_label.shape)
print(one_hot_label.head())

#case 3) torch
                                             #LabelEncoder로 변환한 정수형 라벨(Y)을 사용해야 처리 가능.
y_data = torch.tensor(Y, dtype= torch.int64) #torch는 one_hot할때 정수형 텐서를 입력으로 받아야함. 
                                             #dtype은 정수형을 받아야함(int32는 작동 되지 않고 int64만 작동)
one_ = torch.nn.functional.one_hot(y_data, num_classes=len(e.classes_))

print(one_)

