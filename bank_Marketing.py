# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 13:57:52 2020
@author: Victor
"""
# importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn import metrics

# Reading data and Loading data Frame
df = pd.read_csv("bank Marketing.csv")

# Data Exploration
print(df.head())
# checcking types
print(df.dtypes)

# Encoding Target variable
target = {"deposit":     {"yes": 1, "no": 0}}
df1 = df.replace(target)
df1.head()

# Checking feature Correlations
# applying Pearson Correlation
cor = df.corr()
numeric_df = df1.select_dtypes(exclude="object")
corr_numeric = numeric_df.corr()
sns.heatmap(corr_numeric, annot=True, cmap="RdBu_r")
plt.title("Correlation Matrix", fontsize=16)
plt.show()

# correlation with output variable
cor_deposit = abs(cor["duration"])
#Selecting highly correlated features
imp_features = cor_deposit[cor_deposit>0.4]
print(imp_features)

# assignning new data set
# dropping columns with high correlation to target
df2 = df1.drop(["duration"],axis=1)
print(df2.dtypes)

# Checking of missing value
count_nan = df2.isnull().sum()
print(count_nan[count_nan>0])
#no missing values

# Data visualization
v_counts = df2["deposit"].value_counts()
v_counts
v_plot = df2["deposit"].value_counts().plot(kind='bar')
v_plot
b_plot = df2["education"].value_counts().plot(kind='bar')
b_plot
scatter = df2.plot(kind='scatter',x='education',y='balance')
scatter

# Age distribution
df2.age.hist()
plt.title('Histogram of Age')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.savefig('hist_age')

# Visualising in sweetviz

# importing sweetviz
import sweetviz as sv

#analyzing the dataset
advert_report = sv.analyze(df)

#display the report
advert_report.show_html('bank Marketing.html')

# Encoding Catergorical values
#Labels

labels_df = pd.get_dummies(df2, columns=["job", "marital","education","default","housing","loan","contact","month","poutcome" ])
labels_df.dtypes

# Create new Data Frame with no dtype object
data = labels_df.select_dtypes(exclude="object")
data.head()
print(data.dtypes)

# Splitting data for training and testing
#columns = data.columns.values
#columns
X = data.drop('deposit', axis=1)
y = data['deposit']

# Create training and testing vars
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)


# Models building for Classification

# Decision tree Classifier
from sklearn.tree import DecisionTreeClassifier 
# Create Decision Tree classifer object
dt = DecisionTreeClassifier()
# Train Decision Tree Classifer
dt.fit(X_train, y_train)

#Predict the response for test dataset
DT_pred = dt.predict(X_test)
# model evaluation
# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, DT_pred))
# Classification Report & Confusion matrix
print(confusion_matrix(DT_pred, y_test))
print(classification_report(y_test, DT_pred))


# KNN Classifier
from sklearn.neighbors import KNeighborsClassifier
# KNN model requires you to specify n_neighbors,
# the number of points the classifier will look at to determine what class a new point belongs to

# build model
KNN_model = KNeighborsClassifier(n_neighbors=5)
# fit classifiers
KNN_model.fit(X_train, y_train)
# prediction
KNN_prediction = KNN_model.predict(X_test)
# evaluation
# Accuracy score is the simplest way to evaluate
print("Accuracy:",accuracy_score(KNN_prediction, y_test))
# Classification Report & Confusion matrix
print(confusion_matrix(KNN_prediction, y_test))
print(classification_report(KNN_prediction, y_test))


# SVM Classifier
from sklearn import svm
from sklearn.svm import SVC
# build model
SVC_model = svm.SVC()
# fit classifiers
SVC_model.fit(X_train, y_train)
# prediction
SVC_prediction = SVC_model.predict(X_test)
# Accuracy score 
print("Accuracy:",accuracy_score(SVC_prediction, y_test))
# Classification Report & Confusion matrix
print(confusion_matrix(SVC_prediction, y_test))
print(classification_report(SVC_prediction, y_test))



# Logistic Regression Classifier
from sklearn.linear_model import LogisticRegression
# build model
lg_model = LogisticRegression()
# fit classifiers
lg_model.fit(X_train, y_train)
# prediction
lg_pred = lg_model.predict(X_test)
# Accuracy score
print('Accuracy of logistic regression: {:.2f}'.format(lg_model.score(X_test, y_test)))
# confusion matrix
confusion_matrix = confusion_matrix(y_test, lg_pred)
print(confusion_matrix)
# Classification Report
print(classification_report(y_test, lg_pred))


# Feature importance rank
# import feature selection library
from sklearn.feature_selection import RFE
# Fitting selector objects
predictors = X_train
selector = RFE(lg_model, n_features_to_select= 1)
imp_features = selector.fit(predictors, y_train);

order = imp_features.ranking_
order

feature_ranks = []
for i in order:
    feature_ranks.append(f"{1}. {data.columns[1]}")
feature_ranks

# ROC curve for Log Reg
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
logit_roc_auc = roc_auc_score(y_test, lg_model.predict(X_test))
fpr, tpr, thresholds = roc_curve(y_test, lg_model.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Logistic Regression Classifier ROC curve')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()


# Create a Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier 
#from sklearn.feature_selection import SelectFromModel
rf_model = RandomForestClassifier(n_estimators=100)
# Fitting the classifier
rf_model.fit(X_train, y_train)
RF_pred = rf_model.predict(X_test)
RF_pred

# feature importance in randomforestRandomForestClassfier
rf_model.feature_importances_ 
# Plot of importances
feat_importances = pd.Series(rf_model.feature_importances_, index=X.columns)
feat_importances.nlargest(10).plot(kind='barh')

# Model Accuracy
print("Accuracy:", metrics.accuracy_score(y_test, RF_pred))
# Classification Report & Confusion matrix
print(confusion_matrix(RF_pred, y_test))
print(classification_report(y_test, RF_pred))




