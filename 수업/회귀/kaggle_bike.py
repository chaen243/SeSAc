import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import calendar
import missingno as msno #결측치를 보는 plot


#1. 데이터

path = 'C:\\Users\\r2com\\Desktop\\수업자료\\파일\\bike_shared\\'
df_train = pd.read_csv(path + 'train.csv')
df_test = pd.read_csv(path + 'test.csv')
print(df_test)
# print(df_train)

# print(df_train.describe()) #결측치 없음!
#              season       holiday    workingday       weather         temp         atemp      humidity     windspeed        casual    registered         count
# count  10886.000000  10886.000000  10886.000000  10886.000000  10886.00000  10886.000000  10886.000000  10886.000000  10886.000000  10886.000000  10886.000000
# mean       2.506614      0.028569      0.680875      1.418427     20.23086     23.655084     61.886460     12.799395     36.021955    155.552177    191.574132
# std        1.116174      0.166599      0.466159      0.633839      7.79159      8.474601     19.245033      8.164537     49.960477    151.039033    181.144454
# min        1.000000      0.000000      0.000000      1.000000      0.82000      0.760000      0.000000      0.000000      0.000000      0.000000      1.000000
# 25%        2.000000      0.000000      0.000000      1.000000     13.94000     16.665000     47.000000      7.001500      4.000000     36.000000     42.000000
# 50%        3.000000      0.000000      1.000000      1.000000     20.50000     24.240000     62.000000     12.998000     17.000000    118.000000    145.000000
# 75%        4.000000      0.000000      1.000000      2.000000     26.24000     31.060000     77.000000     16.997900     49.000000    222.000000    284.000000
# max        4.000000      1.000000      1.000000       4.000000     41.00000     45.455000    100.000000     56.996900    367.000000    886.000000    977.000000

# print(df_train.columns) #Index(['datetime', 'season', 'holiday', 'workingday', 'weather', 'temp',
    #    'atemp', 'humidity', 'windspeed', 'casual', 'registered', 'count'],
    #   dtype='object')

# print(df_train.info())    
# print(df_train['datetime'])
# df_train['date'] = df_train.datetime.apply(lambda x: x.split()[0])
# df_train['hour'] = df_train.datetime.apply(lambda x: x.split()[1].split(".")[0])
# df_train['weekday'] = df_train.date.apply(lambda dateString : calendar.day_name[datetime.strptime(dateString,"%Y-%m-%d").weekday()])
# df_train['month'] = df_train.date.apply(lambda dateString : calendar.month_name[datetime.strptime(dateString,"%Y-%m-%d").month])
# df_train['season'] = df_train.season.map({1:"Spring",2:"Summer",3:"Fall",4:"Winter"})
# df_train['weather'] = df_train.weather.map({1: " Clear + Few clouds + Partly cloudy + Partly cludy",\
#                                2 : " Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist ", \
#                                3 : " Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered clouds", \
#                                4 : "Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog"})

#카테고리 유형으로 강제 변환
# categoryVariableList = ['hour','weekday','month','season','weather','holiday','workingday'] 
# for var in categoryVariableList:
#     df_train[var] = df_train[var].astype("category")

msno.matrix(df_train,figsize=(12,5))
# plt.show()    

# df_train['datetime'] = pd.to_datetime(df_train['datetime'])
#년도만 떼어옴                      
# df_train['year'] = df_train['datetime'].dt.year
# df_train['month'] = df_train['datetime'].dt.month
# df_train['day'] = df_train['datetime'].dt.day
# df_train['hour'] = df_train['datetime'].dt.hour
# df_train['minute'] = df_train['datetime'].dt.minute
# df_train['second'] = df_train['datetime'].dt.second
# #요일 데이터 = 일요일은 6 월요일이 0
# df_train['dayofweek'] = df_train['datetime'].dt.dayofweek

# figure, ((ax1,ax2,ax3),(ax4,ax5,ax6)) = plt.subplots(nrows=2,ncols=3)
# figure.set_size_inches(18,8)

# sns.barplot(data = df_train, x='year', y='count',ax= ax1)
# sns.barplot(data = df_train, x='month', y='count',ax= ax2)
# sns.barplot(data = df_train, x='day', y='count',ax= ax3)
# sns.barplot(data = df_train, x='hour', y='count',ax= ax4)
# sns.barplot(data = df_train, x='minute', y='count',ax= ax5)
# sns.barplot(data = df_train, x='second', y='count',ax= ax6)

# ax1.set(ylabel='Count',title='Year rental mount')
# ax2.set(ylabel='month',title='month rental mount')
# ax3.set(ylabel='day',title='day rental mount')
# ax4.set(ylabel='hour',title='hour rental mount')
# # plt.show()

# fig,axes = plt.subplots(nrows=2,ncols=2)
# fig.set_size_inches(12,10)
# sns.boxplot(data = df_train, y='count',orient='v',ax = axes[0][0])
# sns.boxplot(data = df_train, y='count',orient='v',x = 'season',ax = axes[0][1])
# sns.boxplot(data = df_train, y='count',orient='v',x = 'hour',ax = axes[1][0])
# sns.boxplot(data = df_train, y='count',orient='v',x = 'workingday', ax = axes[1][1])

# axes[0][0].set(ylabel='Count',title='Rental amount')
# axes[0][1].set(xlabel='Seoson',ylabel = 'Count',title='Seosonal Rental amount')
# axes[1][0].set(xlabel='Hour of The Day',ylabel = 'Count',title='Hour Rental amount')
# axes[1][1].set(xlabel='Working Day',ylabel = 'Count',title='Working or not Rental amount')
# #이상치와 데이터의 균등성을 유추 가능

# # plt.show()

# fig, (ax1,ax2,ax3,ax4,ax5) = plt.subplots(nrows=5)
# fig.set_size_inches(18,25)

# #꺾은선 그래프
# sns.pointplot(data=df_train, x='hour', y='count',ax=ax1)
# sns.pointplot(data=df_train, x='hour', y='count',hue='workingday',ax=ax2)
# sns.pointplot(data=df_train, x='hour', y='count',hue='dayofweek',ax=ax3)
# sns.pointplot(data=df_train, x='hour', y='count',hue='weather',ax=ax4)
# sns.pointplot(data=df_train, x='hour', y='count',hue='season',ax=ax5)
# # plt.show()

# # corrMatt = df_train.select_dtypes(include=['float64', 'int64']).corr()

# corrMatt = df_train.corr()
# mask = np.array(corrMatt)
# #Return thr undices for upper_triangle of arr.
# #상삼각행렬
# mask[np.tril_indices_from(mask)] = False #False는 하삼각행렬로 나옴

# fig,ax = plt.subplots()
# fig.set_size_inches(20,10)
# sns.heatmap(corrMatt,mask=mask,vmax=0.8,square=True, annot=True)
# # plt.show()


# fig,(ax1,ax2,ax3) = plt.subplots(ncols=3) #windspeed 컬럼에서 0이 너무 많음. 
# #1~5사이는 아예 없어서 이상치 혹은 결측치를 0으로 처리했을 가능성 and 5이하는 0으로 처리했을 가능성이 있음.
# fig.set_size_inches(12,5)
# sns.regplot(x='temp',y='count',data=df_train,ax=ax1)
# sns.regplot(x='windspeed',y='count',data=df_train,ax=ax2)
# sns.regplot(x='humidity',y='count',data=df_train,ax=ax3)
# # plt.show() 

# print(df_train['datetime']) 

#연도&월 나오게 하기
# columns = ['year','month']
# df_train['year_month'] =df_train[columns].apply(lambda row: '-'.join(row.values.astype(str)), axis=1)
# print(df_train['year_month'])

#같은 결과
# def concatenate_year_month(datetime):
#     return "{0}-{1}".format(datetime.year,datetime.month)
# df_train['year_month'] = df_train['datetime'].apply(concatenate_year_month)
# print(df_train['year_month'])    

# fig,(ax1,ax2) = plt.subplots(nrows=1,ncols=2)
# fig.set_size_inches(18,6)

# sns.barplot(data = df_train, x='year', y='count',ax=ax1)
# sns.barplot(data = df_train, x='month', y='count', ax= ax2)


# fig,ax3 = plt.subplots(nrows=1,ncols=1)
# fig.set_size_inches(18,4)
# sns.barplot(data=df_train, x='year_month', y='count', ax=ax3)
# plt.show()


############################이상치##########################
#이상치 처리 (IQR, 3-sigma)
#IQR = Q3(75%)-Q1(25%)
#Q1-1.5*IQR<x,Q3+1.5*IQR

count_q1 = np.percentile(df_train['count'],25)
count_q3 = np.percentile(df_train['count'],75)
count_IQR = count_q3 - count_q1
df_train_IQR = df_train[(df_train['count']>=(count_q1 - (1.5*count_IQR))) &
            (df_train['count']<=(count_q3 + (1.5*count_IQR)))]
print(df_train_IQR) #300개 정도 제거됨

#3sigma 사용 (3sigma 벗어나면 제거)
df_train_sigma = df_train[np.abs(df_train['count'] - df_train['count'].mean() <=3*df_train['count'].std())]
print(df_train_sigma) #100개 정도 제거됨\

#IQR을 적용했을떄의 그림

# fig, axes = plt.subplots(nrows=2, ncols=2)
# fig.set_size_inches(12,10)
# sns.boxplot(data=df_train_IQR, y="count", orient= "v", ax=axes[0][0])
# sns.boxplot(data=df_train_IQR, y="count", x = "season",orient= "v", ax=axes[0][1])
# sns.boxplot(data=df_train_IQR, y="count", x="hour",orient= "v", ax=axes[1][0])
# sns.boxplot(data=df_train_IQR, y="count", x="workingday",orient= "v", ax=axes[1][1])

# axes[0][0].set(ylabel='Count',title="Rental amount")
# axes[0][1].set(xlabel='Season',ylabel='Count',title="Seasonal Rental amount")
# axes[1][0].set(xlabel='Hour of The Day',ylabel='Count',title="Hour Rental amount")
# axes[1][1].set(xlabel='Working Day',ylabel='Count',title="Working or not Rental amount")
# plt.show()



####################합쳐서 처리후 분리하기###########
data = pd.concat([df_train,df_test])
data.reset_index(inplace=True)
data.drop('index', inplace=True, axis=1)

data['date'] = data.datetime.apply(lambda x: x.split()[0])
data['hour'] = data.datetime.apply(lambda x: x.split()[1].split(":")[0]).astype('int')
data['year'] = data.date.apply(lambda x : x.split()[0].split("-")[0])
data['weekday'] = data.date.apply(lambda dateString : datetime.strptime(dateString,"%Y-%m-%d").weekday())
data['month'] = data.date.apply(lambda dateString : datetime.strptime(dateString,"%Y-%m-%d").month)



categoricalFeatureNames = ["season","holiday","workingday","weather","weekday","month","year","hour"]
numericalFeatureNames = ["temp","humidity","windspeed","atemp"]
dropFeatures = ['casual',"count","datetime","date","registered"]

for var in categoricalFeatureNames:
    data[var] = data[var].astype("category")


#data를 붙였다가 나눌때 사용하는 방법 ~는 not
dataTrain = data[pd.notnull(data['count'])].sort_values(by=["datetime"])
dataTest = data[~pd.notnull(data['count'])].sort_values(by=["datetime"]) 
datetimecol = dataTest["datetime"]
yLabels = dataTrain["count"]
yLablesRegistered = dataTrain["registered"]
yLablesCasual = dataTrain["casual"]    

dataTrain  = dataTrain.drop(dropFeatures,axis=1)
dataTest  = dataTest.drop(dropFeatures,axis=1)

def rmsle(y, y_,convertExp=True):
    if convertExp:
        y = np.exp(y),
        y_ = np.exp(y_)
    log1 = np.nan_to_num(np.array([np.log(v + 1) for v in y]))
    log2 = np.nan_to_num(np.array([np.log(v + 1) for v in y_]))
    calc = (log1 - log2) ** 2
    return np.sqrt(np.mean(calc))

#np.nan_to_num : Replace NaN with zero and infinity with large finite numbers (default behaviour)
#or with the numbers defined by the user using the nan, posinf and/or neginf keywords.

np.log(np.NaN)

from sklearn.metrics import mean_squared_error,mean_absolute_error
def rmsle(y,pred):
    log_y = np.log1p(y)
    log_pred = np.log1p(pred)
    squared_error = (log_y-log_pred)**2
    rmsle = np.sqrt(np.mean(squared_error))
    return rmsle
#sklearn의 mean_squared_error 이용해 RMSE계산
def rmse(y,pred):
    return np.sqrt(mean_squared_error(y,pred))


#MSE, RMSE, RMSLE 계산
def evaluate_rgre(y,pred):
    rmsle_val = rmsle(y,pred)
    rmse_val = rmse(y,pred)
    mae_val = mean_absolute_error(y,pred)
    print('RMSLE:{0:.3f}, RMSE:{1:.3f}, MAE:{2:.3f}'.format(rmsle_val,rmse_val,mae_val))


#분리를 통해 추출된 속성은 문자열 속성을 가지고 있음 따라서 숫자형 데이터로 변환해 줄 필요가 있음.
#pandas.to_numeric(): https://pandas.pydata.org/pandas-docs/stable/generated/pandas.to_numeric.html
#coerce : 숫자로 변경된 
dataTrain['year'] = pd.to_numeric(dataTrain.year,errors='coerce')
dataTrain['month'] = pd.to_numeric(dataTrain.month,errors='coerce')
dataTrain['hour'] = pd.to_numeric(dataTrain.hour,errors='coerce')
dataTrain['weekday'] = pd.to_numeric(dataTrain.hour,errors='coerce')

dataTrain['season'] = pd.to_numeric(dataTrain.year,errors='coerce')
dataTrain['holiday'] = pd.to_numeric(dataTrain.month,errors='coerce')
dataTrain['workingday'] = pd.to_numeric(dataTrain.hour,errors='coerce')
dataTrain['weather'] = pd.to_numeric(dataTrain.hour,errors='coerce')

from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
import warnings
pd.options.mode.chained_assignment = None
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Initialize logistic regression model
lModel = LinearRegression()

# Train the model
yLabelsLog = np.log1p(yLabels)
lModel.fit(X = dataTrain,y = yLabelsLog)

# Make predictions
preds = lModel.predict(X= dataTrain)
print ("RMSLE Value For Linear Regression: ",rmsle(np.exp(yLabelsLog),np.exp(preds)))


ridge_m_ = Ridge()
ridge_params_ = { 'max_iter':[3000],'alpha':[0.1, 1, 2, 3, 4, 10, 30,100,200,300,400,800,900,1000]}
rmsle_scorer = metrics.make_scorer(rmsle, greater_is_better=False) #rmsle가 낮아지는 스코어를 알아서 찾아줌.
grid_ridge_m = GridSearchCV( ridge_m_, 
                          ridge_params_,
                          scoring = rmsle_scorer,
                          cv=5)
yLabelsLog = np.log1p(yLabels)
grid_ridge_m.fit( dataTrain, yLabelsLog )
preds = grid_ridge_m.predict(X= dataTrain)
print (grid_ridge_m.best_params_)
print ("RMSLE Value For Ridge Regression: ",rmsle(np.exp(yLabelsLog),np.exp(preds)))

fig,ax= plt.subplots()
fig.set_size_inches(12,5)
df = pd.DataFrame(grid_ridge_m.cv_results_)
df["rmsle"] = df["mean_score_time"].apply(lambda x:-x)
sns.pointplot(data=df,x=df['param_alpha'],y="rmsle",ax=ax)


lasso_m_ = Lasso()

alpha  = 1/np.array([0.1, 1, 2, 3, 4, 10, 30,100,200,300,400,800,900,1000])
lasso_params_ = { 'max_iter':[3000],'alpha':alpha}

grid_lasso_m = GridSearchCV( lasso_m_,lasso_params_,scoring = rmsle_scorer,cv=5)
yLabelsLog = np.log1p(yLabels)
grid_lasso_m.fit( dataTrain, yLabelsLog )
preds = grid_lasso_m.predict(X= dataTrain)
print (grid_lasso_m.best_params_)
print ("RMSLE Value For Lasso Regression: ",rmsle(np.exp(yLabelsLog),np.exp(preds)))

fig,ax= plt.subplots()
fig.set_size_inches(12,5)
df = pd.DataFrame(grid_lasso_m.cv_results_)
df["rmsle"] = df["mean_score_time"].apply(lambda x:-x)
sns.pointplot(data=df,x=df['param_alpha'],y="rmsle",ax=ax)

plt.show()