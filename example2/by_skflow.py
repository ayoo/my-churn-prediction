import skflow
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn import datasets
from sklearn import cross_validation
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
import random

#random.seed(100)

# read and transform features for machine learning friendly ones
def read_dataset(filename):
    used_cols = ['nth_membership_period','Window','Usage_Days','Supercategory_Ratio','Intensity_Ratio',
                 'Product_Article','Overview_Detail','Email_Open_R','Email_Clicked_R','Browser_Researcher', 'Target_Group']
    data = pd.read_csv(filename, sep=',', decimal='.', header=0)[used_cols]
    y = np.where(data['Target_Group'] == 'Renewers', 1.0, 0.0)
    data = data.drop('Target_Group', axis=1)
    feature_names = data.columns
    X = data.fillna(data.mean())

    return X, y, feature_names

def scale_and_split(X, y):
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test

def lin_model(X_train, X_test, y_train, y_test):
    classifier = skflow.TensorFlowLinearClassifier(n_classes=2)
    return classifier.fit(X_train, y_train)

def exp_decay(global_step):
     return tf.train.exponential_decay(
        learning_rate=0.5, global_step=global_step,
         decay_steps=100, decay_rate=0.001)

def dnn_model(X_train, y_train):
    classifier = skflow.TensorFlowDNNClassifier(hidden_units=[3,3],
                                                n_classes=2,
                                                batch_size=50,
                                                learning_rate=exp_decay,
                                                steps=5000)
    return classifier.fit(X_train, y_train)

def evaluate(labels, predictions):
    print metrics.classification_report(labels, predictions)
    print 'AUC score: %f' %  metrics.roc_auc_score(labels, predictions)
    print("Accuracy: %f" % metrics.accuracy_score(labels, predictions))

def main():
    filename = '/Users/andyyoo/scikit_learn_data/renewer/Orange_Dataset.csv'
    X, y, feature_names = read_dataset(filename)
    X_train, X_test, y_train, y_test = scale_and_split(X, y)
    model = dnn_model(X_train, y_train)
    predictions = model.predict(X_test)
    evaluate(y_test, predictions)

if __name__ == '__main__':
    main()

