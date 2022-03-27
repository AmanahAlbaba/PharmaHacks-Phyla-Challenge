import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import classification_report

#read file already converted to percents
data = pd.read_csv("withPercents.csv")

# convert labels to 0,1,2,3
label_encoder_model = LabelEncoder()

y_original_data = data['disease']
y = label_encoder_model.fit_transform(y_original_data)
X = data[data.columns.values.tolist()[1::]]
#Disease-1 is 0
#Disease-2 is 1
#Disease-3 is 2
#Healthy is 3

#remove disease from input data
del X['disease']

#used selector to find top 50 best features
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

#split the training data into 80% training, and 20% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

#assign machine learning algorithm
clf = RandomForestClassifier(n_estimators=300, max_features=40)
clf = clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)

#print report
print(classification_report(y_test, y_pred))
print(cohen_kappa_score(y_test, y_pred))