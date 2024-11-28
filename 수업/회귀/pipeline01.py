#각각의 모델을 합쳐서 쓸때 사용함


from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
import numpy as np

def polynomali_func(X):
    y = 1 + 2 * X + X**2+X**3
    return y
#Pipeline 객체로 streamline하게 Polynopmial feature 변환과 Linear regression을 연결.
model = Pipeline([('Poly', PolynomialFeatures(degree=3)),
                  ('linear', LinearRegression())])
X = np.arange(4).reshape(2,2)
y = polynomali_func(X)

model = model.fit(X, y)
print('Polynomial 회귀 계수 : \n', np.round(model.named_steps['linear'].coef_,2))