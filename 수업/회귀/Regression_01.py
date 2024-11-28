import numpy as np
import matplotlib.pyplot as plt
import random

# np.random.seed(42)
#y = 4x+6 식을 근사, random값은 noise를 위해 만듦.
X = 2*np.random.rand(100,1) #균등분포
y = 6+4*X+np.random.randn(100,1) #정규분포

#X,y 데이터셋 Scatter plot으로 시각화
# plt.scatter(X, y)
# plt.show()

def get_weight_updates(w1, w0, X, y, Learning_rate = 0.01):
    N = len(y) #y = w0+w1*x1 -> 벡터의 길이를 잼.
    w1_update = np.zeros_like(w1) #벡터의 크기에 따라서 0을 mapping 시켜줌.
    w0_update = np.zeros_like(w0) #값이 있어야 답을 바꿀 수 있기 때문에 0을 채워넣음.
    #예측 배열을 계산하고 예측과 실제 값의 차이를 계산.
    y_pred = np.dot(X, w1.T) +w0 #행열 연산을 위해 w1을 transpose 해줌.
    #np.matmul써도 되지만, 현재 코드에서는 벡터 계산이기 때문에 dot을 사용.
    #y=w1*x+w0 -> np.dot(X,w1.T) #w1.T는 N*1을 1*N으로 변환. x -> N*1을 1*1로 변환
    diff = y- y_pred #cost functuion (실제값 - 예측값)

    #w0_update를 행렬 연산으로 구하기 위해 모두 1값을 가진 행렬을 생성.
    w0_factors = np.ones((N, 1)) #초기값 ones로 셋팅 크기를 N의 크기만큼 받아들이고,
    #w1과 w0을 업데이트. w1_update와 w0_update를 계산
    #1로 설정한 이유는 0으로 설정을 하면 y = ax +0은 컴퓨터에서 y = ax 그래프로 인식하기 때문에 1을 넣어서 업데이트를 해줌.
    w1_update = -(2/N)*Learning_rate*(np.dot(X.T,diff)) #summation_i^n (y-y_pred)(-x_i)
    w0_update = -(2/N)*Learning_rate*(np.dot(w0_factors.T,diff))
    return w1_update,w0_update

#입력 인자 iters로 주어진 횟수만큼 반복적으로 w1과 w0에 업데이트를 적용함.
def gradient_descent_steps(X,y,iters=10000):
    #w0과 w1을 모두 0으로 초기화 
    w0 = np.zeros((1,1))
    w1 = np.zeros((1,1))

    #인자로 주어진 iters만큼 반복적으로 get_weight_update()를 호출하여 w1, w0 업데이트를 수행
    for ind in range(iters):
        w1_update,w0_update = get_weight_updates(w1,w0,X,y,Learning_rate=0.01)
        w1 = w1 - w1_update #w1(왼쪽에 있는) -> new, w1(오른쪽에 있는)->old
        #w1_update = gradient_descent방법
        w0 = w0 - w0_update #w1(왼쪽에 있는) -> new, w1(오른쪽에 있는)->old
    return w1, w0

def get_cost(y,y_pred): #y와 y_pred의 손실값 구하기
    N = len(y)
    cost = np.sum(np.square(y-y_pred))/N #y-y_pred의 제곱한 값을 N으로 나눈것을 모두 더함
    return cost  

w1, w0 = gradient_descent_steps(X, y, iters=10000)
#최적의 값을 뽑고 그때의 cost값을 출력
print("W1 : {0:.3f} W0 : {1:.3f}".format(w1[0,0],w0[0,0]))
y_pred = w1[0,0]* X+w0
print('Gradient Descent Total cost:{0:4f}'.format(get_cost(y,y_pred)))

plt.scatter(X, y)
plt.plot(X,y_pred)
plt.show()

############데이터가 많을 때 쓰는 방법############# (batch size)

def stochastic_gradient_descent_steps(X, y, batch_size=10, iters=10000):
    w0 = np.zeros((1,1))
    w1 = np.zeros((1,1))
    prev_cost = 10000
    iter_index = 0
    for ind in range(iters):
        np.random.seed(ind)
        stochastic_random_index = np.random.permutation(X.shape[0]) #순열이 있게 섞어줌
        sample_X = X[stochastic_random_index[0:batch_size]]
        sample_y = y[stochastic_random_index[0:batch_size]]
        #랜덤하게 batch_size만큼 추출된 데이터 기반으로 w1_update,w0_update 계산 후 업데이트
        w1_update, w0_update=get_weight_updates(w1,w0,sample_X,sample_y,Learning_rate=0.01)
        w1 = w1 - w1_update
        w0 = w0 = w0_update
    return w1,w0    

w1,w0 = stochastic_gradient_descent_steps(X,y,iters=1000)
print("w1 :", round(w1[0,0],3), "w0 : ", round(w0[0,0],3))
y_pred = w1[0,0] * X+w0
print('Stochastic Gradient Discent Total Cost:{0: .4f}'.format(get_cost(y,y_pred)))



# 보폭에 따른 결과를 보려고 함.
import numpy as np
import matplotlib.pyplot as plt

lr_list = [0.001, 0.5, 0.3, 0.7]

def get_derivative(lr_list):

  w_old = 2
  derivative = [w_old]

  y = [w_old ** 2] # 손실 함수를 y = x^2로 정의함.

  for i in range(1,10):
    #먼저 해당 위치에서 미분값을 구함

    dev_value = w_old **2

    #위의 값을 이용하여 가중치를 업데이트
    w_new = w_old - lr * dev_value
    w_old = w_new

    derivative.append(w_old) #업데이트 된 가중치를 저장 함,.
    y.append(w_old ** 2) #업데이트 된 가중치의 손실값을 저장 함.

  return derivative, y

x = np.linspace(-2,2,50) 
x_square = [i**2 for i in x]

fig = plt.figure(figsize=(12, 7))

for i,lr in enumerate(lr_list):
  derivative, y =get_derivative(lr)
  ax = fig.add_subplot(2, 2, i+1)
  ax.scatter(derivative, y, color = 'red')
  ax.plot(x, x_square)
  ax.title.set_text('lr = '+str(lr))

plt.show()