import torch

# scalar
s1 = torch.tensor([3,])
s2 = torch.Tensor([3,])
add_scalar_12 = s1 +s2
print(add_scalar_12.size())


# 1-dim
v1 = torch.tensor([1.,2.,3.])
v2 = torch.tensor([4.,5.,6.])
add_vector_12 = v1 + v2
print(add_vector_12.size())


# 2-dim

#gpu 연산방법 (안되면 cpu로 돌아감)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                                            #딥러닝에서는 부동소숫점(float)연산임으로 float으로 형변환 한것.
m1 = torch.randint(0,9,(3,3), device = device, dtype = torch.float)
# m2 = torch.ones_like(m1)

m2 = torch.ones_like(m1)

matrix12 = torch.matmul(m1,m2)

print(matrix12)
