from __future__ import division
import pandas as pd
import numpy as np

import json

from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

print "Importing data"
filename = '/Users/andyyoo/scikit_learn_data/renewer/Orange_Dataset.csv'
used_cols = ['nth_membership_period','Window','Usage_Days','Supercategory_Ratio',
             'Intensity_Ratio', 'Product_Article', 'Overview_Detail', 'Email_Open_R', 'Email_Clicked_R',
             'Browser_Researcher', 'Target_Group_Code']
data = pd.read_csv(filename, sep=',', decimal='.', header=0)[used_cols]
print "Top 10 features with High correlation to the target variable"
print data.corr()['Target_Group_Code'].copy().sort_values(ascending=False)[:10]
y = np.where(data['Target_Group_Code'] == 3, 1, 0)

print "Formatting feature space"
feature_names = data.columns
data = data.drop('Target_Group_Code', axis=1)
data = data.fillna(data.mean())
X = data.as_matrix().astype(np.float)

print "Scaling features"
# This is important but not required for Tree based algorithms like DecisionTree or RandomForest
# scaler = StandardScaler()
# X = scaler.fit_transform(X)

print "Generating training data"
X_train,X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

print "Training classifier using Linear RandomForest Classifier"
clf = RandomForestClassifier(n_estimators=5, max_depth=10,  class_weight='auto')
clf.fit(X_train,y_train)
predictions = clf.predict(X_test)

print '\nY hat:\n',predictions

for e in predictions:
    print e,

print '\nY true:\n',y_test
for e in y_test:
    print e,

print '\n Classification Report:\n'
print metrics.classification_report(y_test, predictions)
print 'AUC score: ', metrics.roc_auc_score(y_test, predictions)
print("Accuracy: %f" % metrics.accuracy_score(y_test, predictions))

