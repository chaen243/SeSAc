import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pylab as plt

cancer = load_breast_cancer()


x_train, x_test, y_train, y_test = train_test_split(cancer.data, cancer.target, stratify=cancer.target, random_state=0)

n_feature = cancer.data.shape[1] #columns에 대한 정보

score_n_tr_est = []
score_n_te_est = []
score_m_tr_mft = []
score_m_te_mft = []

for i in np.arange(1, n_feature+1): # n_estimators와 max_features는 모두 0보다 큰 정수여야 하므로 1부터 시작합니다.

    params_n = {'n_estimators':i, 'max_features':'sqrt', 'n_jobs':-1} # kwargs parameter

    params_m = {'n_estimators':10, 'max_features':i, 'n_jobs':-1}

    forest_n = RandomForestClassifier(**params_n).fit(x_train, y_train)

    forest_m = RandomForestClassifier(**params_m).fit(x_train, y_train)


    score_n_tr = forest_n.score(x_train, y_train)

    score_n_te = forest_n.score(x_test, y_test)

    score_m_tr = forest_m.score(x_train, y_train) 

    score_m_te = forest_m.score(x_test, y_test)



    score_n_tr_est.append(score_n_tr)

    score_n_te_est.append(score_n_te)

    score_m_tr_mft.append(score_m_tr)

    score_m_te_mft.append(score_m_te)


index = np.arange(len(score_n_tr_est))

plt.plot(index, score_n_tr_est, label='n_estimators train score', color='lightblue', ls='--') # ls: linestyle

plt.plot(index, score_m_tr_mft, label='max_features train score', color='orange', ls='--')

plt.plot(index, score_n_te_est, label='n_estimators test score', color='lightblue')

plt.plot(index, score_m_te_mft, label='max_features test score', color='orange')

plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1),

           ncol=2, fancybox=True, shadow=False) # fancybox: 박스모양, shadow: 그림자

plt.xlabel('number of parameter', size=15)

plt.ylabel('score', size=15)

plt.show()