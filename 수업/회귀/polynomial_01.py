from sklearn.preprocessing import PolynomialFeatures
import numpy as np

# X = np.arange(4).reshape(2,2)
# print('일차 단항식 계수 feature:\n', X)

# #일차 단항식 계수 feature:
# #  [[0 1]
# #  [2 3]]

# #degree=2인 2차 다항식으로 변환하기 위해 Polynomial Features를 이용하여 변환.
# poly = PolynomialFeatures(degree=2)
# poly_ftr = poly.fit_transform(X)
# print('변환된 2차 다항식 계수 feature : \n', poly_ftr)

# # 변환된 2차 다항식 계수 feature :
# #  [[1. 0. 1. 0. 0. 1.]
# #  [1. 2. 3. 4. 6. 9.]]

# X = np.arange(4).reshape(2,2)
# print('일차 단항식 계수 feature:\n', X)

# #일차 단항식 계수 feature:
# #  [[0 1]
# #  [2 3]]

# #degree=2인 2차 다항식으로 변환하기 위해 Polynomial Features를 이용하여 변환.
# poly = PolynomialFeatures(degree=3).fit_transform(X)
# # poly_ftr = poly.fit_transform(X)
# print('변환된 3차 다항식 계수 feature : \n', poly)
# print('3차계수 shape', poly.shape)

# # 변환된 3차 다항식 계수 feature :
# #  [[ 1.  0.  1.  0.  0.  1.  0.  0.  0.  1.]
# #  [ 1.  2.  3.  4.  6.  9.  8. 12. 18. 27.]]

# def polynomail_func(x):
#     y = 1 + 2 * X + X **2 + X **3
#     return y
# y = polynomail_func(X)
# #limear regression에 3차 다항식 계수 feature와 3차 다항식 결정값으로 학습 후 회귀 계수 확인
# from sklearn.linear_model import LinearRegression
# model = LinearRegression()
# model.fit(poly, y)
# print('Polynomial 회귀 계수 : \n', np.round(model.coef_,2))
# print('Polynomial 회귀 shape : ', model.coef_.shape)

