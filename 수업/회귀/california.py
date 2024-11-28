from sklearn.datasets import fetch_california_housing
housing = fetch_california_housing(as_frame=True)
housing = housing.frame
#medinc(중위소득), housing age(주택 연식), averooms(평균 방 개수), avebedrms(평균 침실수)
#population(인구수),aveoccup(평균 거주자 수),latitude(위도), longitude(경도)
# print(housing.describe())
# print(housing.info())
# import matplotlib.pyplot as plt
# # housing.hist(bins=50, figsize=(12,8))
# num_colums = len(housing.columns)
# #열의 길이
# fig, axs = plt.subplots(nrows =(num_colums+2)//3, ncols=3, figsize=(15,10))
# #3줄씩 끊어가는것, 1행>3개만 표시 num_columns+2 >0+2,1+2....
# #figsize>(가로 크기, 세로크기)
# axs =axs.flatten() #flatten>벡터로 만들어주는것. 일렬로 늘려쓰는것.
# for i, column in enumerate(housing.columns):
#     housing.boxplot(column = column, ax = axs[i]) #axs로 하나씩 그린것
#     axs[i].set_title(column) #하나씩 그린것에 대해서 title을 달아주는것

# plt.tight_layout() #layout 잡아주세요.
# plt.show()
  
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# #상관관계 분석
# housing = fetch_california_housing()
# data = pd.DataFrame(housing.data, columns = housing.feature_names)
# data['MedHouseVal'] = housing.target
# corr_matrix = data.corr()
# plt.figure(figsize = (12,8))
# sns.heatmap(corr_matrix, annot = True, cmap='coolwarm', fmt='.2f',linewidths=0.5) #히트맵 표시
# plt.title('Corrleation Matrix')
# plt.show()


def iqr_outliers():
    global housing
    outlier_list = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup'] #이상치

    for i, item in enumerate(outlier_list):
        Q1,Q3 = np.percentile(housing[item],[25,75]) #Q1: 하위 25%, Q3: 상위 25%
        IQR = Q3 - Q1
        lower_bound = Q1 -1.5*IQR #IQR 하한선
        upper_bound = Q3 +1.5*IQR #IQR 상한선
        lower_data = housing[item]>=lower_bound
        upper_data = housing[item]<=upper_bound
        outliers = len(housing [(housing[item]<lower_bound) | (housing[item]>upper_bound)])
        print(f'{item}:{outliers}')

        housing =housing[lower_data&upper_data]

housing_before = housing.copy()
iqr_outliers()
housing_after = housing.copy()
import matplotlib.pyplot as plt
# print(f'After Remove Outliers: {housing.shape}')

#plotting
def plot_boxplots(before,after):
    features = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup'] #이상치
    fig, axs = plt.subplots(nrows=2, ncols=len(features),figsize=(18,8),sharey='row')
    fig.suptitle('Boxplots Before and After Outlier Removal', fontsize = 16)

    #plot before removing outliers
    for i, feature in enumerate(features):
        before.boxplot(column=feature, ax= axs[0,i])
        axs[0,i].set_title(f'before: {feature}')
        axs[0,i].tick_params(axis='x',labelrotation=45)
    #plot after removing outliers
    for i, feature in enumerate(features):
        after.boxplot(column=feature, ax = axs[1,i])
        axs[1,i].set_title(f'after: {feature}')
        axs[1,i].tick_params(axis ='x', labelrotation=45)
    plt.tight_layout(rect=[0,0,1,0.96])
    # plt.show()
plot_boxplots(housing_before, housing_after)

from scipy import stats
def z_score_outliers():
    global housing
    features = housing.drop(columns = ['MedHouseVal','Longitude','Latitude'])
    z_scores = np.abs(stats.zscore(features))
    threshold = 3
    outliers = (z_scores > threshold).any(axis=1)
    print('Z-Score Outliers Count : ', outliers.sum())
    housing = housing.loc[~outliers]
z_score_outliers()

from sklearn.model_selection import train_test_split
x = housing.drop(columns=['MedHouseVal']) #입력데이터(특성) - medhouseval을 제외한 모든 특성
y = housing['MedHouseVal']

X_train, X_test, y_train, y_test =train_test_split(x,y,test_size= 0.2,random_state=42)
data_mse = {}
data_score = {}
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

def linear_regression_model():
    model = LinearRegression()
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test,y_pred)
    data_mse['linear_regression'] =mse
    data_score['linear_regression'] = r2_score(y_test,y_pred)
    return y_pred

from sklearn.linear_model import Lasso, Ridge
import optuna
from optuna import Trial
from optuna.samplers import TPESampler
# lasso_data_mse = {}
# lasso_data_score = {}
# def lasso_regression():
#     lasso_reg = Lasso(alpha=0.01)
#     lasso_reg.fit(X_train,y_train)
#     y_pred = lasso_reg.predict(X_test)

#     mse = mean_squared_error(y_test,y_pred)
#     data_mse['lasso_regression'] =mse
#     data_score['lasso_regression'] = r2_score(y_test,y_pred)
#     return y_pred

# ridge_data_mse = {}
# ridge_data_score = {}
def ridge_regression():
    ridge_reg = Ridge(alpha=1.5)
    ridge_reg.fit(X_train,y_train)
    y_pred = ridge_reg.predict(X_test)

    mse = mean_squared_error(y_test,y_pred)
    data_mse['ridge_regression'] =mse
    data_score['ridge_regression'] = r2_score(y_test,y_pred)
    return y_pred

# y_pred =linear_regression_model()
# y_pred_lasso = lasso_regression()

def objectiveRidge(trial : Trial, x, y, test):
    param = {'alpha' : trial.suggest_float('alpha', 0.001, 1.5, log=True)}
    y_pred_ridge = Ridge(**param)
# res_data = pd.DataFrame({'Actural':y_test, 'Predicted': y_pred, 'Predicted_lasso' : y_pred_lasso, 'Predicted_ridge' : y_pred_ridge})   #비교를 위해 dataframe으로 만듬
    res_data = pd.DataFrame({'Actural':y_test, 'Predicted_ridge' : y_pred_ridge})   #비교를 위해 dataframe으로 만듬

# res_data['Error'] = res_data['Actural'] - res_data['Predicted'] #error (실제값 -예측값)
    print(res_data.head(5))
    print("MSE: ", data_mse)
    print('R^2 score', data_score) 

    return data_mse, res_data, param, y_pred_ridge

study = optuna.create_study(direction='minimize',sampler= TPESampler())

study.optimize(lambda trial : objectiveRidge(trial, x, y, X_test), n_trials=500)
# print('Best trial: score {},\nparams {}'.format(study.best_trial.value, study.best_trial.params))

