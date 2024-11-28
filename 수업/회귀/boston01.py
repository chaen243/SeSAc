import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


path = 'C:\\Users\\r2com\\Desktop\\수업자료\\파일\\'
df = pd.read_csv(path + 'boston.csv')
X = df.drop(columns='MEDV')
y = df[['MEDV']]
print(X,y)

import statsmodels.api as sm
X_constant = sm.add_constant(X) #절편이 있음
model_1 = sm.OLS(y,X_constant)
lin_reg = model_1.fit()
print(lin_reg.summary())

#                             OLS Regression Results
# ==============================================================================
# Dep. Variable:                   MEDV   R-squared:                       0.741 #adj r^2를 확인 하는게 좋음
# Model:                            OLS   Adj. R-squared:                  0.734  과적합에 빠져있으면 평균적으로 0.2정도의 차이가 있음
# Method:                 Least Squares   F-statistic:                     108.1 분산
# Date:                Thu, 21 Nov 2024   Prob (F-statistic):          6.72e-135
# Time:                        10:47:16   Log-Likelihood:                -1498.8
# No. Observations:                 506   AIC:                             3026. AIC/BIC는 작을수록 좋음
# Df Residuals:                     492   BIC:                             3085.
# Df Model:                          13
# Covariance Type:            nonrobust

#age/indus = pvalue로는 유의미하지 않지만 모델의 성능은 신뢰할수 있는 상태. 
                                                  #p-value
# ==============================================================================
#                  coef    std err          t      P>|t|      [0.025      0.975]
# ------------------------------------------------------------------------------
# const         36.4595      5.103      7.144      0.000      26.432      46.487
# CRIM          -0.1080      0.033     -3.287      0.001      -0.173      -0.043
# ZN             0.0464      0.014      3.382      0.001       0.019       0.073
# INDUS          0.0206      0.061      0.334      0.738      -0.100       0.141
# CHAS           2.6867      0.862      3.118      0.002       0.994       4.380
# NOX          -17.7666      3.820     -4.651      0.000     -25.272     -10.262
# RM             3.8099      0.418      9.116      0.000       2.989       4.631
# AGE            0.0007      0.013      0.052      0.958      -0.025       0.027
# DIS           -1.4756      0.199     -7.398      0.000      -1.867      -1.084
# RAD            0.3060      0.066      4.613      0.000       0.176       0.436
# TAX           -0.0123      0.004     -3.280      0.001      -0.020      -0.005
# PTRATIO       -0.9527      0.131     -7.283      0.000      -1.210      -0.696
# B              0.0093      0.003      3.467      0.001       0.004       0.015
# LSTAT         -0.5248      0.051    -10.347      0.000      -0.624      -0.425
# ==============================================================================
# Omnibus:                      178.041   Durbin-Watson:                   1.078
# Prob(Omnibus):                  0.000   Jarque-Bera (JB):              783.126
# Skew:#분포의 정도                1.521   Prob(JB):                    8.84e-171
# Kurtosis:                       8.281   Cond. No.                     1.51e+04
# ==============================================================================

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

lr = LinearRegression()
neg_mse_score = cross_val_score(lr,X,y,scoring='neg_mean_squared_error', cv=5)
rmse_scores = np.sqrt(-1*neg_mse_score) #사이킷런의 scoring함수에는 score값이 높을수록 좋은 평가결과라고 자동 인식하기 때문에
                                        #원래의 평가지표에 -1을 곱해서 작은 오류값이 더 큰 숫자로 인식하게 해줌.
avg_rmse = np.mean(rmse_scores)

print('5 folds의 개별 Negative MSE scores:', np.round(neg_mse_score,2))
print('5 folds의 개별 RMSE scores:', np.round(rmse_scores,2))
print('5 folds의 평균 RMSE: {0: .3f}'.format(avg_rmse))

