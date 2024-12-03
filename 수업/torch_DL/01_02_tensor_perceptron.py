import torch

#1. 데이터

x_data = torch.FloatTensor([[1.],[2.],[3,],[4,]])
y_data = torch.FloatTensor([[1.],[3.],[5,],[7,]])

# 랜덤 텐서 생성하기

# torch.manual_seed: 동일한 결과를 만들 도록 seed를 고정한다.
# torch.rand: [0, 1) 사이의 랜덤 텐서 생성
# torch.randn: 평균=0, 표준편차=1 인 정규분포로부터 랜덤 텐서 생성
# torch.randint: [최저값, 최대값) 사이에서 랜덤 정수 텐서 생성

# 평균 0, 분산 1의 파라미터의 정규분포로 부터 값을 가져옴.
# 학습을 통해 업데이트가 되어 변화되는 모델의 파라미터인 w,b를 의미한다.

W = torch.randn(1, 1) #랜덤난수

b = torch.randn(1, 1) #랜덤난수

print("W : ", W)
print("b : ", b)

for j in range(len(x_data)):
    # data * weight 작성
    WX = torch.matmul(x_data[j] ,W) # perceptron 모델
    

    ## bias add 작성
    y_hat = torch.add(WX, b)


    ## W와 b로 예측 하기
    print("y_data: , ",y_data[j], "prediction : ", y_hat)