import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import torch.optim.adam
import torch.optim.sgd
from torch.utils.data import DataLoader, TensorDataset

# 데이터 선언
df = pd.read_csv(r"C:\Users\r2com\Desktop\수업자료\파일\house_price\house_price_of_unit_area.csv")
# print(df.head())

# print(df.info()) #데이터정보


dataset = df.drop(columns=['house price of unit area']).values 
label_data = df['house price of unit area']
# print(label_data)
print(dataset)

dataset = torch.tensor(dataset, dtype= torch.float32)
label_data = torch.tensor(label_data, dtype= torch.float32)

tensordataset = TensorDataset(dataset, label_data)
dataloader = DataLoader(tensordataset, batch_size=100, shuffle=True)

#퍼셉트론 모델 구현
class Model(nn.Module):
    def __init__(self, input_size, output_size):
        super(Model, self).__init__()
        self.layer1 = nn.Linear(input_size, 64)
        self.layer2 = nn.Linear(64, 128)
        self.layer3 = nn.Linear(128, 258)
        self.layer4 = nn.Linear(258, 128)
        self.layer5 = nn.Linear(128, 64)
        self.layer6 = nn.Linear(64, 32)
        self.layer7 = nn.Linear(32, 8)
        self.output_layer = nn.Linear(8, output_size)
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
        x = self.layer6(x)
        x = self.ReLu(x)
        x = self.layer7(x)
        x = self.ReLu(x)
        x = self.output_layer(x)
        return x


#모델 인스턴스 생성
input_size = dataset.shape[1]
output_size = 1
model = Model(input_size, output_size)

criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

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

    print(f'Epoch [{epoch+1}/{epochs}], Loss: {total_loss / len(dataloader):.4f}')

