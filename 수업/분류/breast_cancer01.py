from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split



cancer = load_breast_cancer()

X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, stratify= cancer.target, random_state=42)
#stratify = 정확한 target을 한번 더 지정.

clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train,y_train)
print('Accyracy on training set : {:.3f}'.format(clf.score(X_train, y_train)))
print('Accyracy on test set : {:.3f}'.format(clf.score(X_test, y_test)))


#max_depth만 건들여서 확인
tree = DecisionTreeClassifier(random_state=42, max_depth=5)
tree.fit(X_train,y_train)
print('Accyracy on training set : {:.3f}'.format(tree.score(X_train, y_train)))
print('Accyracy on test set : {:.3f}'.format(tree.score(X_test, y_test)))

from sklearn.tree import export_graphviz
import graphviz




export_graphviz(
    tree,
    out_file="tree.dot",
    class_names=['malignant', 'benign'],
    feature_names=cancer.feature_names,
    impurity=False,
    filled=True
)

# # 그래프 표시
# with open("tree.dot") as f:
#     dot_graph = f.read()

# graphviz.Source(dot_graph).view()


print('Feature importances : ')
print(tree.feature_importances_)

import numpy as np
import matplotlib.pyplot as plt

def plot_feature_importances_cancer(model):
    n_features = cancer.data.shape[1]
    plt.barh(np.arange(n_features), model.feature_importances_,align = 'center')
    plt.yticks(np.arange(n_features), cancer.feature_names)
    plt.xlabel('Feature importance')
    plt.ylabel('Feature')
    plt.ylim(-1, n_features)

plot_feature_importances_cancer(tree)
plt.show()    
