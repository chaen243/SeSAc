import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# data 선언
x_data = torch.FloatTensor([[1.],[2.],[3.],[4.]])
y_data = torch.FloatTensor([[1.],[3.],[5.],[7.]])

dataset = TensorDataset(x_data,y_data) #x,y를 하나의 데이터셋으로 묶음(x,y를 매칭시키기 위해)
batchsize = 1
data_loader = DataLoader(dataset= dataset, batch_size= batchsize, shuffle=True) #배치사이즈만큼 모델에 넣기 위해 이용
                                                                                #shuffle은 데이터의 순서도 학습할수 있기때문에 이용

class LinearRegression(nn.Module):
    def __init__(self):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(1,1)
    def forward(self, x):
        return self.linear(x)

model = LinearRegression()

# cost & optimizer
criterion = torch.nn.MSELoss() #손실함수
optimizer = torch.optim.SGD(model.parameters(), lr= 0.001) #가중치 업데이트 하는 방법 정의

epochs = 2000
for epoch in range(epochs):
    model.train() #모델을 학습모드로 설정
    for x_batch, y_batch in data_loader: #배치사이즈만큼 돌려줌
        optimizer.zero_grad() # 동적신경망이므로 초기화 해야함. 에포마다 초기화
        y_pred = model(x_batch) #입력데이터를 모델에 통과해 예측값 계산
        # loss function
        loss = criterion(y_pred,y_batch) #예측값,실제값 손실을 계산
        loss.backward() #역전파로 그래디언트 계산
        optimizer.step() #파라미터 업데이트

    if epoch % 100 == 0:
        print(f'Epoch {epoch +1}/{epochs}, Loss: {loss.item()}') #loss.item (현재 손실값)


from torchsummary import summary

summary(model, (1,1))    

model.eval() #가중치를 추적하지 않겠다. eval/ with문 같이 써야함
with torch.no_grad(): #with문에서는 미분을 추적하지 않겠다고 막아둠. (predict만 할거라 미분을 할 필요가 없음)
    predict = model(x_data)
    predict = predict.cpu().data.numpy() #gpu에 있는 predict를 cpu로 내림
    print('train:', x_data)
    print('predict:', predict)

for param_tensor in model.state_dict(): #model이 훈련이 끝난 결과들은 state_dict에 저장해둠
                                        #훈련이 끝난 모델의 weight만 가져가서 작업하면 같은 결과를 내줌.
    print(param_tensor, "\t", model.state_dict()[param_tensor])   

#     ----------------------------------------------------------------
#         Layer (type)               Output Shape         Param #
# ================================================================
#             Linear-1                 [-1, 1, 1]               2  #outputshape -1 = batchsize (어떤 정수라도 가능.)