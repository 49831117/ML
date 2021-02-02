from sklearn.datasets import load_iris
from sklearn import tree

X, y = load_iris(return_X_y=True)

'''
iris = load_iris()
X = iris.data       # 觀察值
y = iris.target     # 標記值
'''

clf = tree.DecisionTreeClassifier()
# https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html?highlight=decisiontreeclassifier#sklearn.tree.DecisionTreeClassifier

clf = clf.fit(X, y)
tree.plot_tree(clf)