import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix


train = pd.read_csv("train.csv").drop(["Id","Behaviour"],axis=1)
test = pd.read_csv("test.csv").drop(["Id","Behaviour"],axis=1)

train = train.drop_duplicates(keep='first')

X_train = train.drop('Attrition',axis=1).values
y_train = train["Attrition"].values

X_test = test.values

 
#Preprocessing of Training Data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
col = [1,2,5,8,10,12,15]

for i in col:
    label_X = LabelEncoder()
    X_train[:,i] = label_X.fit_transform(X_train[:,i])
    
from sklearn.compose import ColumnTransformer
ct=ColumnTransformer(transformers=[('encode',OneHotEncoder(drop="first"),[1,2,5,10,12])],remainder='passthrough')
X_train = ct.fit_transform(X_train)
Z = pd.DataFrame(X_train)

#Preprocessing of Testing Data
for i in col:
    label_X = LabelEncoder()
    X_test[:,i] = label_X.fit_transform(X_test[:,i])

from sklearn.compose import ColumnTransformer
ct1=ColumnTransformer(transformers=[('encode',OneHotEncoder(drop="first"),[1,2,5,10,12])],remainder='passthrough')
X_test = ct1.fit_transform(X_test)
Z1 = pd.DataFrame(X_test)   

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Applying Kernel PCA
from sklearn.decomposition import KernelPCA
kpca = KernelPCA(n_components = 29, kernel = 'rbf')
X_train = kpca.fit_transform(X_train)
X_test = kpca.transform(X_test)

#LOGISTIC REGRESSION

from sklearn.linear_model import LogisticRegression
logistic = LogisticRegression(C=1.5,solver='liblinear',penalty='l1',random_state = 0)
logistic.fit(X_train, y_train)

#Prediciting Training Set Results
y_pred_train_logistic = logistic.predict(X_train)
y_pred_train_logistic_proba = logistic.predict_proba(X_train)[:,1]

print(roc_auc_score(y_train, y_pred_train_logistic_proba))


cm_logistic = confusion_matrix(y_train, y_pred_train_logistic)

# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies_logistic = cross_val_score(estimator = logistic, X = X_train, y = y_train, cv = 10)
print(accuracies_logistic.mean())
print(accuracies_logistic.std())

# Applying Grid Search to find the best model and the best parameters
from sklearn.model_selection import GridSearchCV
parameters = [ {'C': [0.1,0.01,1.0,1.5], 'solver': ['newton-cg']},
               {'C': [0.1,0.01,1.0,1.5], 'solver': ['lbfgs']},
               {'C': [0.1,0.01,1.0,1.5], 'solver': ['liblinear'],'penalty':['l1']},
               {'C': [0.1,0.01,1.0,1.5], 'solver': ['liblinear'],'penalty':['l2']},
               {'C': [0.1,0.01,1.0,1.5], 'solver': ['saga'],'penalty':['l1']},
               {'C': [0.1,0.01,1.0,1.5], 'solver': ['saga'],'penalty':['l1']}]
grid_search_logistic = GridSearchCV(estimator = logistic,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10,
                           n_jobs = -1)
grid_search_logistic = grid_search_logistic.fit(X_train, y_train)
best_accuracy_logistic = grid_search_logistic.best_score_
best_parameters_logistic = grid_search_logistic.best_params_


#Prediciting Test Set Results
y_pred_test_logistic_proba = logistic.predict_proba(X_test)[:,1]

#Compiling into CSV file
list_logistic = list(zip(pd.read_csv("test.csv").Id, y_pred_test_logistic_proba))
df = pd.DataFrame(data=list_logistic,columns=['Id','Attrition'])
df.to_csv('logistic_KPCA.csv',index=False)



# Fitting K-NN to the Training set
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
knn.fit(X_train, y_train)

# Predicting the Test set results
y_pred_train_knn = knn.predict(X_train)
y_pred_train_knn_proba = knn.predict_proba(X_train)[:,1]

print(roc_auc_score(y_train, y_pred_train_knn_proba))

cm_knn = confusion_matrix(y_train, y_pred_train_knn)

accuracies_knn = cross_val_score(estimator = knn, X = X_train, y = y_train, cv = 10)
print(accuracies_knn.mean())
print(accuracies_knn.std())

#Prediciting Test Set Results
y_pred_test_knn_proba = knn.predict_proba(X_test)[:,1]
list_knn = list(zip(pd.read_csv("test.csv").Id, y_pred_test_knn_proba))
df1 = pd.DataFrame(data=list_knn,columns=['Id','Attrition'])
df1.to_csv('knn_KPCA.csv',index=False)



# Fitting SVM to the Training set
from sklearn.svm import SVC
svc = SVC( probability= True,random_state = 0)
svc.fit(X_train, y_train)

# Predicting the Training set results
y_pred_train_svc = svc.predict(X_train)
y_pred_train_svc_proba = svc.predict_proba(X_train)[:,1]


#AUC-ROC Score of Training Set
print(roc_auc_score(y_train, y_pred_train_svc_proba))

cm_svc = confusion_matrix(y_train, y_pred_train_svc)

accuracies_svc = cross_val_score(estimator = svc, X = X_train, y = y_train, cv = 10)
print(accuracies_svc.mean())
print(accuracies_svc.std())

# Predicting the Test set results
y_pred_test_svc_proba = svc.predict_proba(X_test)[:,1]


#Compiling into CSV file
list_svc = list(zip(pd.read_csv("test.csv").Id, y_pred_test_svc_proba))
df2 = pd.DataFrame(data=list_svc,columns=['Id','Attrition'])
df2.to_csv('svc_KPCA.csv',index=False)


from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(n_estimators = 10, criterion = 'gini', random_state = 0,max_features='log2')
forest.fit(X_train, y_train)

# Predicting the Test set results
y_pred_train_forest = forest.predict(X_train)
y_pred_train_forest_proba = forest.predict_proba(X_train)[:,1]

cm_forest = confusion_matrix(y_train, y_pred_train_forest)

print(roc_auc_score(y_train, y_pred_train_forest_proba))

accuracies_forest = cross_val_score(estimator = forest, X = X_train, y = y_train, cv = 10)
print(accuracies_forest.mean())
print(accuracies_forest.std())

# Predicting the Test set results
y_pred_test_forest_proba = forest.predict_proba(X_test)[:,1]

#Compiling into CSV file
list_forest = list(zip(pd.read_csv("test.csv").Id, y_pred_test_forest_proba))
df3 = pd.DataFrame(data=list_forest,columns=['Id','Attrition'])
df3.to_csv('forest_KPCA.csv',index=False)

parameters_forest = [ {'n_estimators': [10,9,8,7,6,5], 'criterion': ['entropy'],'max_features':['auto']},
                      {'n_estimators': [10,9,8,7,6,5], 'criterion': ['gini'],'max_features':['auto']},
                      {'n_estimators': [10,9,8,7,6,5], 'criterion': ['entropy'],'max_features':['sqrt']},
                      {'n_estimators': [10,9,8,7,6,5], 'criterion': ['gini'],'max_features':['sqrt']},
                      {'n_estimators': [10,9,8,7,6,5], 'criterion': ['entropy'],'max_features':['log2']},
                      {'n_estimators': [10,9,8,7,6,5], 'criterion': ['gini'],'max_features':['log2']}]
grid_search_forest = GridSearchCV(estimator = forest,
                           param_grid = parameters_forest,
                           scoring = 'accuracy',
                           cv = 10,
                           n_jobs = -1)
grid_search_forest = grid_search_forest.fit(X_train, y_train)
best_accuracy_forest = grid_search_forest.best_score_
best_parameters_forest = grid_search_forest.best_params_






