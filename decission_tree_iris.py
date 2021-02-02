from sklearn.datasets import load_iris
import seaborn as sns
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.tree import export_text
import pickle



# 1. 資料觀察
newX = [[3.7, 2.6, 3.2, 2.2],[3.1, 3.2, 4.8, 1.8]]
iris = sns.load_dataset('iris')
iris.head()
sns.set()
sns.pairplot(iris, hue='species', height=3)

# 2. 讀進資料，建立模型
iris = load_iris()
X = iris.data
y = iris.target
clf = tree.DecisionTreeClassifier()

# 3. 進行cross validation
clf = clf.fit(X, y)
score = cross_val_score(clf, X, y, cv=10, scoring='accuracy')
print(score)
print("%0.2f accuracy with a standard deviation of %0.2f" % (score.mean(), score.std()))

# 4. 列印錯差矩陣(confusion matrix)
y_pred = cross_val_predict(clf, X, y, cv=10)
conf_mat = confusion_matrix(y, y_pred)
print(confusion_matrix(y, y_pred))

# 5. 列印錯差矩陣的性能指標
print(classification_report(y, y_pred))

# 6. 列印決策樹
for i in range(len(X)):
    print(X[i], y_pred[i])

# 7. 列印決策規則
tree_rules = export_text(clf, feature_names=iris['feature_names'])
print(tree_rules)

# 8. 預測新案例的分類結果：



# newX = [[7.7, 2.6, 3.2, 2.2],[3.1, 3.2, 4.8, 1.8]]
# print(clf.predict(newX))
# tree_rules = export_text(clf, feature_names=X)


# pkl_filename = "iris_model.pkl"
# with open(clf, 'wb') as file: #寫二進制文件
# pickle.dump(clf, file)
# with open(clf, 'rb') as file: #讀二進制文件
# pickle_model = pickle.load(file)
# newX = [[7.7, 2.6, 3.2, 2.2],[3.1, 3.2, 4.8, 1.8]]
# print(pickle_model.predict(newX))

# newX = [[7.7, 2.6, 3.2, 2.2],[3.1, 3.2, 4.8, 1.8]]
# print(clf.predict(newX))
# tree_rules = export_text(clf, feature_names=X)