
import seaborn as sns 
import matplotlib.pyplot as plt

iris = sns.load_dataset('iris')
iris.head()
sns.set()
sns.pairplot(iris, hue='species', height=3)