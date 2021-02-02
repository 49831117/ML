# DECISIONTREEREGRESSOR

from sklearn.datasets import load_iris
from sklearn import svm
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, classification_report

iris = load_iris()
X = iris.data
y = iris.target

clf = svm.SVC(kernel='linear', gamma=10)
clf.fit(X, y)
scores = cross_val_score(clf, X, y, cv=10, scoring='accuracy')

print(scores)
print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))

y_pred = clf.predict(X_test)
conf_mat = confusion_matrix(y, y_pred)


print(confusion_matrix(y, y_pred))
print(classification_report(y, y_pred))