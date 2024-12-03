import torch
import torch.nn as nn
import torch.optim as optim

# 데이터 선언
x_data = torch.tensor([[0.,0.], [0.,1.], [1.,0.],[1.,1.]], dtype=torch.float32)
y_data = torch.tensor([[0.], [1.], [1.], [0.]], dtype=torch.float32)


# 퍼셉트론 모델 구현
class MultiLayerPerceptron(nn.Module):
    def __init__(self):
        super(MultiLayerPerceptron, self).__init__()
        self.layer1 = nn.Linear(2, 16)
        self.layer2 = nn.Linear(16, 32)
        self.layer3 = nn.Linear(32, 16)
        self.layer4 = nn.Linear(16, 8)
        self.layer5 = nn.Linear(8, 4)
        self.output_layer = nn.Linear(4, 1)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        x = self.layer1(x)
        x = self.sigmoid(x)
        x = self.layer2(x)
        x = self.sigmoid(x)
        x = self.layer3(x)
        x = self.sigmoid(x)
        x = self.layer4(x)
        x = self.sigmoid(x)
        x = self.layer5(x)
        x = self.sigmoid(x)        
        x = self.output_layer(x)
        x = self.sigmoid(x)
        return x
    
# 모델 인스턴스 생성
model = MultiLayerPerceptron()

# 손실 함수 및 옵티마이저 정의
optimizer = optim.SGD(model.parameters(), lr = 0.05)
criterion = nn.BCELoss()

# 모델 학습
epochs = 2000
for epoch in range(epochs):
    # Forward pass
    outputs = model(x_data)
    loss = criterion(outputs, y_data)

    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    acc = (torch.argmax(outputs, dim=1) == y_data).float().sum()

    # if (epoch+1) % 200 == 0:
    print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# 테스트 데이터 준비
test_data = torch.tensor([[0.5,0.5]], dtype=torch.float32)

# 모델을 통해 테스트 데이터 예측
with torch.no_grad():  # Gradient 계산을 수행하지 않음
    predictions = model(test_data)
    print("Test Data 예측 값:")
    for i, test_val in enumerate(test_data, start=1):
        print(f" test data {test_val.numpy()} 예측 값 : {(predictions>0.5).float()}")    
            