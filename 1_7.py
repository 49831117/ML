# KFOLD CROSS VALIDATION
# MLp3 P.27

from sklearn.datasets import load_iris
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import KFold
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

clf = tree.DecisionTreeClassifier()
kf = KFold(n_splits=2)
kf.get_n_splits(X)
print(kf)
for train_index, test_index in kf.split(X):
    print("TRAIN:", train_index)
    print("TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    print("TRAIN data:")
    print(X_train, y_train)
    print("TEST data:")
    print(X_test, y_test )
