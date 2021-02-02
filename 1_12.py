import pandas as pd
from sklearn import svm
from sklearn.preprocessing import LabelEncoder
from skLearn.metrics import confussion_matrix, classification_report

names = ['age','sex','region','income','married','children','car','save_act','current_act','mortgage','pep']
dataset = pd.read_set
