import pandas as pd
import sys
import seaborn as sns

import numpy as np
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import ExtraTreesClassifier
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
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import classification_report
from sklearn import tree
from sklearn.tree import export_graphviz
from sklearn.metrics import f1_score

from sklearn.feature_selection import SelectKBest,mutual_info_regression
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

#from mlxtend.plotting import plot_decision_regions
np.set_printoptions(threshold=sys.maxsize)
data = pd.read_csv("challenge_1_gut_microbiome_data.csv")
# 2 is positive, 1 is negative, 0 is not related
# look at type of change and type of thing
#data.describe()
#sns.pairplot(data, vars=["Bacteria-1", "Bacteria-2", "Bacteria-3", "Bacteria-4"], hue="disease")
#plt.show()
#print(data)
#data.to_numpy()
#print(data)
maxRange = 1095
maxSamples = 7482
sums = []
for i, row in data.iterrows():
    if (i == maxSamples):
        break
    sum = 0
    for j in range(1,maxRange):
        sum += (row['Bacteria-'+str(j)])
    sums.append(sum)

for i, row in data.iterrows():
    if (i == maxSamples):
        break
    for j in range(1,maxRange):
        row['Bacteria-'+str(j)] = (row['Bacteria-'+str(j)])/sums[i]
        data.iat[i,j] = row['Bacteria-' + str(j)]


# convert labels to 1,2,3
label_encoder_model = LabelEncoder()

y_original_data = data['disease']
y = label_encoder_model.fit_transform(y_original_data)

X = data[data.columns.values.tolist()[1::]]
X
#Disease-1 is 0
#Disease-2 is 1
#Disease-3 is 2
#Healthy is 3

#or i in range(1,7400):
    #print("Row "+str(i+1)+": " + str(y[i-1]))
del X['disease']
#selector = SelectKBest(mutual_info_regression, k =50)
#selector.fit(X, y)
#print(X.columns[selector.get_support()])
goodBacteria = ['Bacteria-41', 'Bacteria-53', 'Bacteria-197', 'Bacteria-233',
       'Bacteria-243', 'Bacteria-408', 'Bacteria-431', 'Bacteria-452',
       'Bacteria-453', 'Bacteria-456', 'Bacteria-505', 'Bacteria-522',
       'Bacteria-530', 'Bacteria-531', 'Bacteria-533', 'Bacteria-536',
       'Bacteria-547', 'Bacteria-548', 'Bacteria-550', 'Bacteria-554',
       'Bacteria-579', 'Bacteria-580', 'Bacteria-581', 'Bacteria-584',
       'Bacteria-598', 'Bacteria-605', 'Bacteria-610', 'Bacteria-615',
       'Bacteria-634', 'Bacteria-635', 'Bacteria-636', 'Bacteria-638',
       'Bacteria-643', 'Bacteria-644', 'Bacteria-648', 'Bacteria-650',
       'Bacteria-653', 'Bacteria-656', 'Bacteria-675', 'Bacteria-688',
       'Bacteria-689', 'Bacteria-696', 'Bacteria-712', 'Bacteria-718',
       'Bacteria-949', 'Bacteria-985', 'Bacteria-989', 'Bacteria-1008',
       'Bacteria-1074', 'Bacteria-1087']
X = X[goodBacteria]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

clf = ExtraTreesClassifier(n_estimators=100, random_state=0)
clf = clf.fit(X_train,y_train)

y_pred = clf.predict(X_test)

print(classification_report(y_test, y_pred))
print(cohen_kappa_score(y_test, y_pred))