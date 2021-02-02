from sklearn.datasets import load_iris
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_predict
import pickle


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

y_pred = cross_val_predict(clf, X, y, cv=10)
conf_mat = confusion_matrix(y, y_pred)

print(confusion_matrix(y, y_pred))
print(classification_report(y, y_pred))

pkl_filename = "iris_model.pkl"
with open(pkl_filename, 'wb') as file: #寫二進制文件
    pickle.dump(iris, file)
with open(pkl_filename, 'rb') as file: #讀二進制文件
    pickle_model = pickle.load(file)
newX = [[7.7, 2.6, 3.2, 2.2],[3.1, 3.2, 4.8, 1.8]]
print(pickle_model.predict(newX))