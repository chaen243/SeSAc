import torch
import torch.nn as nn

# data 선언
x_data = torch.FloatTensor([[0.,0.], [0.,1.], [1.,0.],[1.,1.]])
y_data = torch.FloatTensor([[0.], [1.], [1.], [1.]])
test_data = torch.FloatTensor([[0.8, 0.8]])

input_size = 2
output_size = 1

class BinaryClassification(nn.Module):
    def __init__(self, input_size, output_size):
        super(BinaryClassification, self).__init__()
        self.linear = nn.Linear(2,1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        x = self.linear(x)
        x = self.sigmoid(x)
        return x
    
model = BinaryClassification(input_size=2, output_size=1)
print('model is', model)

# BinaryClassification에 사용되는 loss는 크게 BCELoss와 BCEWithLogitsLoss가 있다.

# BCELoss는 마지막 layer가 sigmoid 혹은 softmax를 사용해 확률값으로 구성 된 경우 사용
# BCEWithLogitsLoss는 확률값으로 변환하지 않더라도 계산 가능

# cost & optimizer
criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

for step in range(5000):  # 에폭
    optimizer.zero_grad()
    # Forward
    y_hat = model(x_data) 
    loss = criterion(y_hat,y_data)
    # Backpropagation
    loss.backward()
    optimizer.step()             # 동적신경망이므로 초기화 해야함.
    ##
    ##

    print("epoch: ", step+1, "error : ", loss.item())

model.eval()
with torch.no_grad():
    # 예측값 계산
    y_hat = model(x_data)
    y_pred = torch.sigmoid(y_hat)
    y_pred_binary = (y_pred >= 0.5).float()

    predict = y_pred.cpu().data.numpy()

    acc = (y_pred_binary == y_data).float().mean().item()    
    print(f'accuracy: {acc * 100: .2f}%')   