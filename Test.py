import pandas as pd
import seaborn as sns

import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix

from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV

from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import plot_roc_curve
from sklearn.metrics import classification_report

from sklearn import tree
from sklearn.tree import export_graphviz
from sklearn.metrics import f1_score

from sklearn.feature_selection import SelectKBest,mutual_info_regression
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

#from mlxtend.plotting import plot_decision_regions

data = pd.read_csv("test.csv")

num = 0
print(data["sentence"][num])
print(data["entities"][num])
print(data["entities_spans"][num])