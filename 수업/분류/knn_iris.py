import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report

#데이터
iris = load_iris()
# print(iris)
# print(iris.keys())
# print(iris.feature_names) #['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
print(iris.target)
#[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#  0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
#  1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2
#  2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
#  2 2]

X_train,X_test,y_train,y_test = train_test_split(iris.data,iris.target,
                                                 test_size=0.2, random_state=42)
# print(X_train.shape) #(120, 4)
# print(X_test.shape) #(30, 4)
# print(y_train.shape) #(120,)
# print(y_test.shape) #(30,)

for i in (1,3,5,7): #k의 갯수
    for j in ('uniform','distance'): #거리계산
        for k in ('auto','ball_tree','kd_tree','brute'): #algorithm
            model = KNeighborsClassifier(n_neighbors=i, weights=j, algorithm=k)
            model.fit(X_train,y_train)
            y_predict = model.predict(X_test)
            relation_square = model.score(X_test,y_test)
            knn_matrix = confusion_matrix(y_test, y_predict)
            print(knn_matrix)
            target_names = ['setosa','versicolor','verginica'] 
            knn_result = classification_report(y_test,y_predict,target_names=target_names)
            print(knn_result)
            print('accuracy : {:.2f}'.format(model.score(X_test,y_test)))                        
        print('\n')
    print('\n')


#               precision    recall  f1-score   support (샘플수)

#       setosa       1.00      1.00      1.00        10
#   versicolor       1.00      1.00      1.00         9
#    verginica       1.00      1.00      1.00        11

#     accuracy                           1.00        30
#    macro avg       1.00      1.00      1.00        30 #대부분 다중분류용
# weighted avg       1.00      1.00      1.00        30