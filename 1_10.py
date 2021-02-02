# DECISIONTREEREGRESSOR

from sklearn.datasets import load_iris
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
# k = 5 for KNeighborsClassifier
iris = load_iris()
X = iris.data
y = iris.target

clf = tree.DecisionTreeRegressor()
clf = clf.fit(X_train, y_train)
scores = cross_val_score(clf, X, y, cv=10, scoring='accuracy')

print(scores)
print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))

y_pred = clf.predict(X_test)
conf_mat = confusion_matrix(y, y_pred)

print(y_pred)
print(confusion_matrix(y, y_pred))
print(classification_report(y, y_pred))