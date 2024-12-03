import torch
import torch.nn as nn
import torch.optim as optim

# data 선언
x_data = torch.FloatTensor([[1.],[2.],[3.],[4.]])
y_data = torch.FloatTensor([[1.],[3.],[5.],[7.]])

# # 1-layer perceptron 만들기 (레이어 1단)
linear = torch.nn.Linear(1,1, bias= True)
model = torch.nn.Sequential(linear)




# cost & optimizer
criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr= 0.001)

for step in range(2000):  # 에폭
    optimizer.zero_grad() # 동적신경망이므로 초기화 해야함.
    y_hat = model(x_data)

    # loss function
    loss = criterion(y_hat,y_data)
    loss.backward()
    optimizer.step()

    print("epoch: ", step, "error : ", loss.item())

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