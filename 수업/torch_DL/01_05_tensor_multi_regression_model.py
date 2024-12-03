import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

#data선언
x_data = torch.FloatTensor([[2.,0.,7.],[6.,4.,2.],[5.,2.,4.],[8.,4.,1.]])
y_data = torch.FloatTensor([[75.],[95.],[91.],[97.]])
test_data = torch.FloatTensor([[5.,5.,7.]])

dataset = TensorDataset(x_data,y_data)
batchsize = 4
data_loader = DataLoader(dataset= dataset, batch_size= batchsize, shuffle=True)

class MultiRegression(nn.Module):
    def __init__(self, input_size, output_size):
        super(MultiRegression, self).__init__()
        self.linear = nn.Linear(input_size, output_size)
    def forward(self, x):
        return self.linear(x)

model = MultiRegression(input_size=3, output_size=1)

# cost & optimizer
criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

epochs = 10000
for epoch in range(epochs):
    model.train()
    for x_batch, y_batch in data_loader:
        optimizer.zero_grad()
        y_pred = model(x_batch)
        #loss function
        loss = criterion(y_pred, y_batch)
        loss.backward()
        optimizer.step()
    if epoch % 100 == 0:
        print(f'Epoch {epoch +1}/{epochs}, Loss: {loss.item()}')

from torchsummary import summary

summary(model, (3,))    

model.eval() #가중치를 추적하지 않겠다. eval/ with문 같이 써야함
with torch.no_grad(): #with문에서는 미분을 추적하지 않겠다고 막아둠. (predict만 할거라 미분을 할 필요가 없음)
    predict = model(x_data)
    predict = predict.cpu().data.numpy() #gpu에 있는 predict를 cpu로 내림
    predict_t = model(test_data)
    predict_t = predict_t.cpu().data.numpy() #gpu에 있는 predict를 cpu로 내림

    print('train:', x_data)
    print('predict:', predict)
    print('predict:', predict_t)
    

for param_tensor in model.state_dict(): #model이 훈련이 끝난 결과들은 state_dict에 저장해둠
                                        #훈련이 끝난 모델의 weight만 가져가서 작업하면 같은 결과를 내줌.
    print(param_tensor, "\t", model.state_dict()[param_tensor])               