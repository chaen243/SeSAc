import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset

# 데이터 로드 및 복사
df = pd.read_csv(r"C:\Users\r2com\Desktop\수업자료\파일\house_price\house_price_of_unit_area.csv")

# 데이터 분리 및 통계 계산
dataset = df.drop(columns=['house price of unit area'])
label_data = df['house price of unit area']
dataset_stats = dataset.describe().T

# 정규화 함수 정의
def min_max_norm(x, stats):
    return (x - stats['min'].values) / (stats['max'].values - stats['min'].values)

def standard_norm(x, stats):
    return (x - stats['mean'].values) / stats['std'].values

# 정규화 수행
normed_train_data = standard_norm(dataset, dataset_stats)

# 텐서 변환
normed_train_data = torch.tensor(normed_train_data.values, dtype=torch.float32)
label_data = torch.tensor(label_data.values, dtype=torch.float32).unsqueeze(1)

# TensorDataset 및 DataLoader 생성
tensordataset = TensorDataset(normed_train_data, label_data)
dataloader = DataLoader(tensordataset, batch_size=100, shuffle=True)

# 모델 정의
class Model(nn.Module):
    def __init__(self, input_size, output_size):
        super(Model, self).__init__()
        self.layer1 = nn.Linear(input_size, 50)
        self.layer2 = nn.Linear(50, 100)
        self.layer3 = nn.Linear(100, 300)
        self.layer4 = nn.Linear(300, 100)
        self.layer5 = nn.Linear(100, 50)
        self.output_layer = nn.Linear(50, output_size)
        self.ReLu = nn.ReLU()

    def forward(self, x):
        x = self.ReLu(self.layer1(x))
        x = self.ReLu(self.layer2(x))
        x = self.ReLu(self.layer3(x))
        x = self.ReLu(self.layer4(x))
        x = self.ReLu(self.layer5(x))
        x = self.output_layer(x)
        return x

# 모델 생성
input_size = normed_train_data.shape[1]
output_size = 1
model = Model(input_size, output_size)

# 손실 함수 및 옵티마이저 설정
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 학습 루프
epochs = 2000
for epoch in range(epochs):
    total_loss = 0
    for batch_input, batch_label in dataloader:
        outputs = model(batch_input)
        loss = criterion(outputs, batch_label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    if (epoch + 1) % 200 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {total_loss / len(dataloader):.4f}')