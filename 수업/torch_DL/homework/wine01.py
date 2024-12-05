import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler





df = pd.read_csv(r'C:\Users\r2com\Desktop\수업자료\파일\winequality-red.csv')
# print(df.head())

# print(df.columns)
# Index(['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
#        'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
#        'pH', 'sulphates', 'alcohol', 'quality'],
#       dtype='object')

print(df['quality'].unique()) #[5 6 7 4 8 3]



dataset = df.copy()

X = dataset.drop(columns=['quality']).values
Y = dataset['quality'].values

le = LabelEncoder()
Y_trans = le.fit_transform(Y)
Y_trans = torch.tensor(Y_trans, dtype= torch.int64)

print(Y_trans)

# StandardScaler 사용
scaler = StandardScaler()
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)  

# 다시 PyTorch 텐서로 변환
X_scaled = torch.tensor(X_scaled, dtype=torch.float32)


X_train1, X_test, Y_train1, Y_test = train_test_split(X_scaled, Y_trans, test_size=0.2, shuffle=True, random_state=123)  ## shuffle=True로 하면 데이터를 섞어서 나눔
## 학습 셋에서 학습과 검증 데이터(0.2)로 구분
X_train, X_valid, Y_train, Y_valid = train_test_split(X_train1, Y_train1, test_size=0.2, shuffle=True, random_state=123)  ## shuffle=True로 하면 데이터를 섞어서 나눔

# print(X_train.shape) #(96, 4)
# print(Y_train.shape) #(96, 3)
# print(X_test.shape) #(30, 4)
# print(Y_test.shape) #(30, 3)
# print(X_valid.shape) #(24, 4)
# print(Y_valid.shape) #(24, 3)

# NumPy 배열을 PyTorch 텐서로 변환
X_train = torch.tensor(X_train, dtype=torch.float32)
X_valid = torch.tensor(X_valid, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)

# DataLoader 생성
train_dataset = TensorDataset(X_train, Y_train)
valid_dataset = TensorDataset(X_valid, Y_valid)
test_dataset = TensorDataset(X_test, Y_test)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=True)

class classificationModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(classificationModel, self).__init__()
        self.layer1 = nn.Linear(input_size, 55)
        self.layer2 = nn.Linear(55, 110)
        self.layer3 = nn.Linear(110, 220)
        self.layer4 = nn.Linear(220, 110)
        self.layer5 = nn.Linear(110, 55)
        # self.layer6 = nn.Linear(64, 32)
        # self.layer7 = nn.Linear(32, 8)
        self.output_layer = nn.Linear(55, output_size)
        self.ReLU = nn.ReLU()

    def forward(self, x):
        x = self.layer1(x)
        x = self.ReLU(self.layer2(x))
        x = self.layer3(x)
        x = self.ReLU(self.layer4(x))
        x = self.ReLU(self.layer5(x))
        # x = self.ReLU(self.layer6(x))
        # x = self.ReLU(self.layer7(x))
        x = self.output_layer(x)  # Softmax는 CrossEntropyLoss에서 자동 적용됨
        return x

    
input_size = X.shape[1]  
output_size = len(le.classes_)



model = classificationModel(input_size, output_size)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 모델 훈련
num_epochs = 1000
train_losses = []
valid_losses = []

#훈련 for문
for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(train_loader)  # 배치 수로 나누어 평균 계산

    # Validation loss
    model.eval()
    valid_loss = 0
    with torch.no_grad():
        for inputs, targets in valid_loader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            valid_loss += loss.item()
        valid_loss /= len(valid_loader)

    if (epoch+1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Training Loss: {train_loss:.4f}, Validation Loss: {valid_loss:.4f}')
    # loss function을 그리기 위해 변수에 값들 저장
    train_losses.append(train_loss)
    valid_losses.append(valid_loss)

# 테스트 데이터로 모델 평가
model.eval()
test_loss = 0
correct = 0
total = 0
with torch.no_grad():
    for inputs, targets in test_loader:
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        test_loss += loss.item()
        
        # 정확도 계산
        _, predicted = torch.max(outputs, 1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()

test_loss /= len(test_loader)
accuracy = 100 * correct / total

print(f"\nTest Loss: {test_loss:.4f}, Test Accuracy: {accuracy:.2f}%")
