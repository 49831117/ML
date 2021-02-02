# train and test
# MLp3 P.26

from sklearn.datasets import load_iris
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
# k = 5 for KNeighborsClassifier
iris = load_iris()
X = iris.data
y = iris.target
clf = tree.DecisionTreeClassifier()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42) 
clf = clf.fit(X_train, y_train)
print(clf.score(X_test, y_test))
y_pred = clf.predict(X_test)
print(y_pred)