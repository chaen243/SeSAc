import torch
import torch.nn as nn
import torch.optim as optim

# 데이터 선언
x_data = torch.tensor([[5.], [30.], [95.], [100.], [265.], [270.], [290.], [300.], [365.]], dtype=torch.float32)
y_data = torch.tensor([[0.], [0.], [0.], [0.], [1.], [1.], [1.], [1.], [1.]], dtype=torch.float32)

# 퍼셉트론 모델 구현
class PerceptronModel(nn.Module):
    def __init__(self):
        super(PerceptronModel, self).__init__()
        self.linear = nn.Linear(1,1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        x = self.linear(x)
        x = self.sigmoid(x)
        return x
    
model = PerceptronModel()
print('model is', model)   
# 손실 함수 및 옵티마이저 정의
criterion = torch.nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)


    # 모델 학습
epochs = 2000
for epoch in range(epochs):
    # Forward pass
    outputs = model(x_data)
    loss = criterion(outputs,y_data)

    # Backward and optimize
    loss.backward()
    optimizer.step()  
    #
    #
    y_pred = torch.sigmoid(outputs)
    y_pred_binary = (y_pred >= 0.5).float()

    acc = acc = (y_pred_binary == y_data).float().mean().item()   

    # if (epoch+1) % 200 == 0:
    print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}, Acc: {acc}')

# 테스트 데이터 준비
test_data = torch.tensor([[7.], [80.], [110.], [180.], [320.]], dtype=torch.float32)

# 모델을 통해 테스트 데이터 예측
with torch.no_grad():  # Gradient 계산을 수행하지 않음
    predictions = model(test_data)
    print("Test Data 예측 값:")
    for i, test_val in enumerate(test_data, start=1):
        print(f" test data {test_val.item()} 예측 값 : {predictions[i-1].item()}")    