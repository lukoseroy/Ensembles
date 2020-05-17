#!/usr/bin/env python
# coding: utf-8

# In[8]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import tree
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from imblearn.over_sampling import SMOTE  # imblearn library can be installed using pip install imblearn
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier


# In[9]:


# Importing dataset and examining it
dataset = pd.read_csv("E-Shop.csv")
pd.set_option('display.max_columns', None) # Will ensure that all columns are displayed
print(dataset.head())
print(dataset.shape)
print(dataset.info())
print(dataset.describe())


# In[3]:


# Converting Categorical features into Numerical features
def converter(column):
    if column == True:
        return 1
    else:
        return 0

dataset['Weekend'] = dataset['Weekend'].apply(converter)
dataset['Transaction'] = dataset['Transaction'].apply(converter)

print(dataset.info())


# In[4]:


dataset['VisitorType'] = dataset['VisitorType'].map({'Returning_Visitor':1, 'New_Visitor':0})
dataset['Month'] = dataset['Month'].map({'Feb':1, 'Mar':2, 'May':3,'June':4, 'Jul':5, 'Aug':6,'Sep':7,'Oct':8,'Nov':9,'Dec':10})


# In[5]:


dataset.head(2)


# In[6]:


# Dividing dataset into label and feature sets
X = dataset.drop('Transaction', axis = 1) # Features
Y = dataset['Transaction'] # Labels
print(type(X))
print(type(Y))
print(X.shape)
print(Y.shape)


# In[7]:


# Normalizing numerical features so that each feature has mean 0 and variance 1
feature_scaler = StandardScaler()
X_scaled = feature_scaler.fit_transform(X)
X_scaled


# In[8]:


# Dividing dataset into training and test sets
X_train, X_test, Y_train, Y_test = train_test_split( X_scaled, Y, test_size = 0.3, random_state = 100)

print(X_train.shape)
print(X_test.shape)


# In[9]:


# Implementing Oversampling to balance the dataset; SMOTE stands for Synthetic Minority Oversampling Technique
print("Number of observations in each class before oversampling (training data): \n", pd.Series(Y_train).value_counts())

smote = SMOTE(random_state = 101)
X_train,Y_train = smote.fit_sample(X_train,Y_train)

print("Number of observations in each class after oversampling (training data): \n", pd.Series(Y_train).value_counts())


# In[10]:


np.unique(Y)


# In[11]:


labels = np.unique(Y); print(labels)


# In[12]:


pd.Series(dataset['Transaction']).value_counts()


# In[13]:


# Tuning the random forest parameter 'n_estimators' and implementing cross-validation using Grid Search
rfc = RandomForestClassifier(criterion='entropy', max_features='auto', random_state=1)
grid_param = {'n_estimators': [200, 250, 300, 350, 400, 450]}

gd_sr = GridSearchCV(estimator=rfc, param_grid=grid_param, scoring='recall', cv=5)

"""
In the above GridSearchCV(), scoring parameter should be set as follows:
scoring = 'accuracy' when you want to maximize prediction accuracy
scoring = 'recall' when you want to minimize false negatives
scoring = 'precision' when you want to minimize false positives
scoring = 'f1' when you want to balance false positives and false negatives (place equal emphasis on minimizing both)
"""

gd_sr.fit(X_train, Y_train)

best_parameters = gd_sr.best_params_
print(best_parameters)

best_result = gd_sr.best_score_ # Mean cross-validated score of the best_estimator
print(best_result)


# In[14]:


# Building random forest using the tuned parameter
rfc = RandomForestClassifier(n_estimators=200, criterion='entropy', max_features='auto', random_state=1)
rfc.fit(X_train,Y_train)
featimp = pd.Series(rfc.feature_importances_, index=list(X)).sort_values(ascending=False)
print(featimp)

Y_pred = rfc.predict(X_test)
print('Classification report: \n', metrics.classification_report(Y_test, Y_pred))

conf_mat = metrics.confusion_matrix(Y_test, Y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(conf_mat,annot=True)
plt.title("Confusion_matrix")
plt.xlabel("Predicted Class")
plt.ylabel("Actual class")
plt.show()
print('Confusion matrix: \n', conf_mat)
print('TP: ', conf_mat[1,1])
print('TN: ', conf_mat[0,0])
print('FP: ', conf_mat[0,1])
print('FN: ', conf_mat[1,0])


# In[15]:


# Selecting features with higher sifnificance and building random forest
X1 = dataset[['PageValue', 'Month', 'ExitRate', 'ProductRelated_Duration', 'ProductRelated']]

feature_scaler = StandardScaler()
X1_scaled = feature_scaler.fit_transform(X1)

X1_train, X1_test, Y1_train, Y1_test = train_test_split( X1_scaled, Y, test_size = 0.3, random_state = 100)

smote = SMOTE(random_state = 101)
X1_train,Y1_train = smote.fit_sample(X1_train,Y1_train)

rfc = RandomForestClassifier(n_estimators=450, criterion='entropy', max_features='auto', random_state=1)
rfc.fit(X1_train,Y1_train)

Y_pred = rfc.predict(X1_test)
print('Classification report: \n', metrics.classification_report(Y1_test, Y_pred))

conf_mat = metrics.confusion_matrix(Y1_test, Y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(conf_mat,annot=True)
plt.title("Confusion_matrix")
plt.xlabel("Predicted Class")
plt.ylabel("Actual class")
plt.show()
print('Confusion matrix: \n', conf_mat)
print('TP: ', conf_mat[1,1])
print('TN: ', conf_mat[0,0])
print('FP: ', conf_mat[0,1])
print('FN: ', conf_mat[1,0])


# In[17]:


# Tuning the AdaBoost parameter 'n_estimators' and implementing cross-validation using Grid Search
abc = AdaBoostClassifier(random_state=1)
grid_param = {'n_estimators': [5,10,20,30,40,50,60,70]}

gd_sr = GridSearchCV(estimator=abc, param_grid=grid_param, scoring='recall', cv=5)

"""
In the above GridSearchCV(), scoring parameter should be set as follows:
scoring = 'accuracy' when you want to maximize prediction accuracy
scoring = 'recall' when you want to minimize false negatives
scoring = 'precision' when you want to minimize false positives
scoring = 'f1' when you want to balance false positives and false negatives (place equal emphasis on minimizing both)
"""

gd_sr.fit(X_train, Y_train)

best_parameters = gd_sr.best_params_
print(best_parameters)

best_result = gd_sr.best_score_ # Mean cross-validated score of the best_estimator
print(best_result)


# In[18]:


# Building AdaBoost using the tuned parameter
abc = AdaBoostClassifier(n_estimators=60, random_state=1)
abc.fit(X_train,Y_train)
featimp = pd.Series(abc.feature_importances_, index=list(X)).sort_values(ascending=False)
print(featimp)

Y_pred = abc.predict(X_test)
print('Classification report: \n', metrics.classification_report(Y_test, Y_pred))

conf_mat = metrics.confusion_matrix(Y_test, Y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(conf_mat,annot=True)
plt.title("Confusion_matrix")
plt.xlabel("Predicted Class")
plt.ylabel("Actual class")
plt.show()
print('Confusion matrix: \n', conf_mat)
print('TP: ', conf_mat[1,1])
print('TN: ', conf_mat[0,0])
print('FP: ', conf_mat[0,1])
print('FN: ', conf_mat[1,0])


# In[24]:


# Tuning the Gradent Boost parameter 'n_estimators' and implementing cross-validation using Grid Search
gbc = GradientBoostingClassifier(random_state=1)
grid_param = {'n_estimators': [10,20,30,40,50,60,70], 'max_depth' : [5,6,7,8,9,10,11,12], 'max_leaf_nodes': [8,12,16,20,24,28,32]}

gd_sr = GridSearchCV(estimator=gbc, param_grid=grid_param, scoring='recall', cv=5)

"""
In the above GridSearchCV(), scoring parameter should be set as follows:
scoring = 'accuracy' when you want to maximize prediction accuracy
scoring = 'recall' when you want to minimize false negatives
scoring = 'precision' when you want to minimize false positives
scoring = 'f1' when you want to balance false positives and false negatives (place equal emphasis on minimizing both)
"""

gd_sr.fit(X_train, Y_train)

best_parameters = gd_sr.best_params_
print(best_parameters)

best_result = gd_sr.best_score_ # Mean cross-validated score of the best_estimator
print(best_result)


# In[25]:


# Building Gradient Boost using the tuned parameter
gbc = GradientBoostingClassifier(n_estimators=20, max_depth=11, max_leaf_nodes=32, random_state=1)
gbc.fit(X_train,Y_train)
featimp = pd.Series(gbc.feature_importances_, index=list(X)).sort_values(ascending=False)
print(featimp)

Y_pred = gbc.predict(X_test)
print('Classification report: \n', metrics.classification_report(Y_test, Y_pred))

conf_mat = metrics.confusion_matrix(Y_test, Y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(conf_mat,annot=True)
plt.title("Confusion_matrix")
plt.xlabel("Predicted Class")
plt.ylabel("Actual class")
plt.show()
print('Confusion matrix: \n', conf_mat)
print('TP: ', conf_mat[1,1])
print('TN: ', conf_mat[0,0])
print('FP: ', conf_mat[0,1])
print('FN: ', conf_mat[1,0])


# In[6]:


import seaborn as sns


# In[11]:


#Checking how many have subscribed w.r.t balance
sns.boxplot(x='Transaction',y='PageValue', data=dataset)


# In[13]:


#Checking how many have subscribed w.r.t balance
sns.barplot(x='Transaction',y='Month', data=dataset)


# In[14]:


#Checking how many have subscribed w.r.t balance
sns.barplot(x='Transaction',y='Administrative', data=dataset)


# In[15]:


#Checking how many have subscribed w.r.t balance
sns.barplot(x='Transaction',y='ProductRelated', data=dataset)


# In[18]:


#Checking how many have subscribed w.r.t balance
sns.barplot(x='Month',y='ProductRelated', data=dataset)


# In[19]:


#Checking how many have subscribed w.r.t balance
sns.barplot(x='Transaction',y='BounceRate', data=dataset)


# In[20]:


#Checking how many have subscribed w.r.t balance
sns.barplot(x='Transaction',y='ExitRate', data=dataset)

