import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()

#data 확인!
# print(type(cancer)) #<class 'sklearn.utils._bunch.Bunch'> 사이킷런의 고유한 특징.

# print(dir(cancer)) #['DESCR', 'data', 'data_module', 'feature_names', 'filename', 'frame', 'target', 'target_names']

# print(cancer.data.shape) #(569, 30)

# print(cancer.feature_names) 
# ['mean radius' 'mean texture' 'mean perimeter' 'mean area'
#  'mean smoothness' 'mean compactness' 'mean concavity'
#  'mean concave points' 'mean symmetry' 'mean fractal dimension'
#  'radius error' 'texture error' 'perimeter error' 'area error'
#  'smoothness error' 'compactness error' 'concavity error'
#  'concave points error' 'symmetry error' 'fractal dimension error'
#  'worst radius' 'worst texture' 'worst perimeter' 'worst area'
#  'worst smoothness' 'worst compactness' 'worst concavity'
#  'worst concave points' 'worst symmetry' 'worst fractal dimension']

# print(cancer.target_names) #['malignant' 'benign']

# print(cancer.target)

#갯수 세어줌
print(np.bincount(cancer.target)) #[212 357]

for i,name in enumerate(cancer.feature_names):
    print('%02d: %s' %(i,name))
malignant = cancer.data[cancer.target==0]
benign = cancer.data[cancer.target==1]

_, bin = np.histogram(cancer.data[:,0],bins=20)
print(np.histogram(cancer.data[:,0], bins=20))

# plt.hist(malignant[:,0], bins= bin, alpha=0.3)
# plt.hist(benign[:,0], bins= bin, alpha=0.3)
# plt.title(cancer.feature_names[0])
# plt.show()

# plt.figure(figsize=(20,15))
# for col in range(30):
#     plt.subplot(8,4,col+1)
#     _,bins = np.histogram(cancer.data[:,col],bins=20)
#     plt.hist(malignant[:,col],bins=bins, alpha=0.3)
#     plt.hist(benign[:,col],bins=bins, alpha=0.3)
#     plt.title(cancer.feature_names[col])
#     if col==0:plt.legend(cancer.target_names)
#     plt.xticks([])
# plt.show()

# fig = plt.figure(figsize=(14,14))
# fig.suptitle('Breast Cancer - Feature analysis', fontsize=20)
# for col in range(cancer.feature_names.shape[0]):
#     plt.subplot(8,4,col+1)
#     plt.scatter(cancer.data[:,0], cancer.target, c= cancer.target, alpha=0.5)
#     plt.title(cancer.feature_names[col]+ ('(%d)' %col))
#     plt.xticks([])
#     plt.yticks([])
# plt.show()

from sklearn.linear_model import LogisticRegression #분!류!모!델

scores = []
 
# #k-fold와 비슷하지만 train과 test 데이터가 겹치는지 안겹치는지 확인할수 없으므로 kfold를 쓰는게 더 좋음! 
# for i in range(10):
#     X_train, X_test, y_train, y_test = train_test_split(cancer.data,  cancer.target)

#     model = LogisticRegression()
#     model.fit(X_train, y_train)

#     score = model.score(X_test, y_test)
#     scores.append(score)
# print('scorese = ',scores)    

flg,axes = plt.subplots(5,6,figsize= [12,20])
flg.suptitle('mean radius VS others', fontsize=20)

for i in range(30):
    ax = axes.flatten()[i]
    ax.scatter(cancer.data[:,0], cancer.data[:,i], c = cancer.target, cmap= 'winter', alpha=0.1)
    ax.set_title(cancer.feature_names[i]+ ('\n(%d)' %i))
    ax.set_axis_off
plt.show()                    