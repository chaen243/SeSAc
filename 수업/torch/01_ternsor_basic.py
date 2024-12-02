import torch

# torch.add, torch.sub, torch.mul, torch.div를 통해 변수를 선언하고 ((4*2)-(1+2)) - 5 를 계산해주세요!

A = torch.tensor(4)
B = torch.tensor(2)
C = torch.tensor(1)
D = torch.tensor(2)
E = torch.tensor(5)

#동일한 값
# A = torch.Tensor([4])  #tensor = [4,]  Tensor= [0.0,0.0,0.0,0.0]
# B = torch.Tensor([2])
# C = torch.Tensor([1]) 
# D = torch.Tensor([2])
# E = torch.Tensor([5])


# 1줄에 torch함수 하나씩만 사용하세요!
out1 = torch.mul(A,B)
out2 = torch.add(C,D)
out3 = torch.sub(out1,out2)

output = torch.sub(out3,5)

print("result = {}".format(output))
