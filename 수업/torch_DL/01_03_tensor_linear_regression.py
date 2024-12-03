import torch

## data 선언
x_data = torch.FloatTensor([[1.],[2.],[3.],[4.]])
y_data = torch.FloatTensor([[1.],[3.],[5.],[7.]])

# 평균 0, 분산 1의 파라미터의 정규분포로 부터 값을 가져옴.
# 학습을 통해 업데이트가 되어 변화되는 모델의 파라미터인 w,b를 의미한다.

W = torch.nn.Parameter(torch.normal(mean=0, std=1, size=(1,1))) #size = step이 1이기때문에 사이즈도 1,1로 만듬
b = torch.nn.Parameter(torch.normal(mean=0, std=1, size=(1,1))) #parameter = w,b를 자동으로 parameter로 추적을 해 학습할 대상으로 인지.
lr = torch.tensor(0.005)
print("W : ", W) #requires_grad=True는 미분업데이트에 추적이 될 대상이라는 뜻.
print("b : ", b)
print('lr : ', lr)

for i in range(8000):  ## 에폭 (훈련 횟수)
    total_error = 0

    for j in range(len(x_data)): ## 배치 1 (배치사이즈) 실제 데이터는 사이즈가 굉장히 크기때문에 데이터를 잘라서 넣어주는것.
        ## data * weight
        WX = torch.matmul(x_data[j] ,W)
        # (1, 1) * (1, 1)

        ## bias add
        y_hat = torch.add(WX, b)
        ######################################여기까지가 forward

        ## 정답인 Y와 출력값의 error 계산
        error = torch.sub(y_data[j],y_hat)

        ## 경사하강법으로 W와 b 업데이트.
        ## 도함수 구하기
        diff_W = torch.matmul(error, x_data[j])
        diff_b = error

        ##  업데이트할 만큼 러닝레이트 곱
        diff_W = torch.multiply(lr, diff_W)
        diff_b = torch.multiply(lr, diff_b)

        ## w, b 업데이트
        W = torch.add(W, diff_W)
        b = torch.add(b, diff_b)

        ## 토탈 에러.
        visual_error = torch.square(error)
        total_error = total_error - visual_error

    ## 모든 데이터에 따른 error 값
    print("epoch: ", i, "error : ", total_error/len(x_data))

 #inference
print(torch.add(torch.multiply(10, W), b))   