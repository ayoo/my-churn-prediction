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
churn_df = pd.read_csv('churn.csv')

print "Formatting feature space"
churn_result = churn_df['Churn?']
y = np.where(churn_result == 'True.',1,0)
to_drop = ['State','Area Code','Phone','Churn?']
churn_feat_space = churn_df.drop(to_drop,axis=1)
yes_no_cols = ["Int'l Plan","VMail Plan"]
churn_feat_space[yes_no_cols] = churn_feat_space[yes_no_cols] == 'yes'
features = churn_feat_space.columns
X = churn_feat_space.as_matrix().astype(np.float)

print "Scaling features"
# This is important but not required for Tree based algorithms like DecisionTree or RandomForest
# scaler = StandardScaler()
# X = scaler.fit_transform(X)

print "Generating training data"
train_index,test_index = train_test_split(churn_df.index)

test_churn_df = churn_df.ix[test_index]
test_churn_df.to_csv("test_churn.csv")

print "Training classifier using Linear RandomForest Classifier"
clf = RandomForestClassifier(n_estimators=15, max_depth=10)  #SVC(probability=True)
clf.fit(X[train_index],y[train_index])
predictions = clf.predict(X[test_index])

print '\nY hat:\n',predictions
print '\nY true:\n',y[test_index]

print metrics.classification_report(y[test_index], predictions)
print 'AUC score: ', metrics.roc_auc_score(y[test_index], predictions)
print("Accuracy: %f" % metrics.accuracy_score(y[test_index], predictions))

'''
             precision    recall  f1-score   support

          0       0.95      0.99      0.97       720
          1       0.89      0.70      0.78       114

avg / total       0.95      0.95      0.94       834

AUC score:  0.843932748538
Accuracy: 0.947242

'''