import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim


import pandas as pd
import torch

df = pd.read_csv(r'C:\Users\r2com\Desktop\수업자료\파일\iris.csv', names = ["sepal_length", "sepal_width", "petal_length", "petal_width", "species"])
dataset=df.copy()

Y= df['species'].values
X= df.drop(columns=['species']).values

df.head()

# y_data = torch.tensor(Y, dtype= torch.int64)

# 문자열을 숫자로 변환
e = LabelEncoder()
e.fit(Y)
Y_trans = e.transform(Y)
Y_trans = torch.tensor(Y_trans, dtype = torch.int64)

# 전체 데이터에서 학습 데이터와 테스트 데이터(0.2)로 구분
X_train1, X_test, Y_train1, Y_test = train_test_split(X, Y_trans, test_size=0.2, shuffle=True, random_state=123)  ## shuffle=True로 하면 데이터를 섞어서 나눔
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

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

class classificationModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(classificationModel, self).__init__()
        self.layer1 = nn.Linear(input_size, 50)
        self.layer2 = nn.Linear(50, 100)
        self.layer3 = nn.Linear(100, 300)
        self.layer4 = nn.Linear(300, 100)
        self.layer5 = nn.Linear(100, 50)
        # self.layer6 = nn.Linear(64, 32)
        # self.layer7 = nn.Linear(32, 8)
        self.output_layer = nn.Linear(50, output_size)
        self.ReLU = nn.ReLU()

    def forward(self, x):
        x = self.ReLU(self.layer1(x))
        x = self.ReLU(self.layer2(x))
        x = self.ReLU(self.layer3(x))
        x = self.ReLU(self.layer4(x))
        x = self.ReLU(self.layer5(x))
        # x = self.ReLU(self.layer6(x))
        # x = self.ReLU(self.layer7(x))
        x = self.output_layer(x)  # Softmax는 CrossEntropyLoss에서 자동 적용됨
        return x

    
input_size = X.shape[1]  
output_size = len(e.classes_)

model = classificationModel(input_size, output_size)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# 모델 훈련
num_epochs = 2000
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

    # Validation loss
    model.eval()
    valid_loss = 0
    with torch.no_grad():
        for inputs, targets in valid_loader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            valid_loss += loss.item()

    if (epoch+1) % 200 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Training Loss: {train_loss:.4f}, Validation Loss: {valid_loss:.4f}')
    # loss function을 그리기 위해 변수에 값들 저장
    train_losses.append(train_loss)
    valid_losses.append(valid_loss)

# 테스트 데이터로 모델 평가
model.eval()
test_loss = 0
with torch.no_grad():
    for inputs, targets in test_loader:
        outputs = model(inputs)
        test_loss = criterion(outputs, targets)
        test_loss += loss.item()
test_loss /= len(test_loader)        
print("\n Test acc: %.4f" % torch.sqrt(test_loss).item())

# 손실 그래프 그리기
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Training loss')
plt.plot(valid_losses, label='Validation loss')
plt.title('Training and Validation Loss')
plt.legend()            
plt.show()

