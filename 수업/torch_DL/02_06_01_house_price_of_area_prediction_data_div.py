import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

#데이터 불러오기
df = pd.read_csv(r"C:\Users\r2com\Desktop\수업자료\파일\house_price\house_price_of_unit_area.csv")

#불러와졌는지 확인
df.head()

#x,y 데이터로 분할
x_data = df.drop(columns= ['house price of unit area']).values
y_data = df['house price of unit area'].values

# PyTorch Tensors로 변환 (torch는 tensor데이터로만 돌아감)
x_data = torch.tensor(x_data, dtype= torch.float32)
y_data = torch.tensor(y_data, dtype= torch.float32).unsqueeze(1)

#데이터 분할
X_train1, X_test, Y_train1, Y_test = train_test_split(x_data, y_data, test_size= 0.2, random_state=123)
X_train, X_valid, Y_train, Y_valid = train_test_split(X_train1, Y_train1, test_size=0.2, random_state=123)

# DataLoader 생성
train_dataset = TensorDataset(X_train, Y_train)
valid_dataset = TensorDataset(X_valid, Y_valid)
test_dataset = TensorDataset(X_test,Y_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

# 모델 정의
class RegressionModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(RegressionModel, self).__init__()
        self.layer1 = nn.Linear(input_size, 50)
        self.layer2 = nn.Linear(50, 100)
        self.layer3 = nn.Linear(100, 300)
        self.layer4 = nn.Linear(300, 100)
        self.layer5 = nn.Linear(100, 50)
        # self.layer6 = nn.Linear(64, 32)
        # self.layer7 = nn.Linear(32, 8)
        self.output_layer = nn.Linear(50, output_size)
        self.ReLu = nn.ReLU()

    def forward(self, x):
        x = self.layer1(x)
        x = self.ReLu(x)    
        x = self.layer2(x)
        x = self.ReLu(x)
        x = self.layer3(x)
        x = self.ReLu(x)
        x = self.layer4(x)
        x = self.ReLu(x)
        x = self.layer5(x)
        x = self.ReLu(x)
        # x = self.layer6(x)
        # x = self.ReLu(x)
        # x = self.layer7(x)
        # x = self.ReLu(x)
        x = self.output_layer(x)
        return x

#데이터 사이즈 지정
input_size = x_data.shape[1]
output_size = 1    
#모델
model = RegressionModel(input_size, output_size)

# 손실 함수와 옵티마이저
criterion = nn.MSELoss()
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
print("\n Test RMSE: %.4f" % torch.sqrt(test_loss).item())

# 손실 그래프 그리기
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Training loss')
plt.plot(valid_losses, label='Validation loss')
plt.title('Training and Validation Loss')
plt.legend()            
plt.show()