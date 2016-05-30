import skflow
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn import cross_validation
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
import random

random.seed(100)

# read and transform features for machine learning friendly ones
def read_dataset(filename):
    drop_cols = ['State','Area Code','Phone','Churn?']
    yes_no_cols = ["Int'l Plan","VMail Plan"]
    churn_data = pd.read_csv(filename, sep=',', decimal='.', header=0)
    churn_data[yes_no_cols] = churn_data[yes_no_cols] == 'yes'
    churn_data['Churn?'] = churn_data['Churn?'] == 'True.'
    y = churn_data['Churn?'].astype(np.float32)
    churn_data = churn_data.drop(drop_cols, axis=1)
    feature_names = churn_data.columns
    X = churn_data.as_matrix().astype(np.float32)
    return X, y, feature_names

def scale_and_split(X, y):
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,y, test_size=0.2)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test

def lin_model(X_train, X_test, y_train, y_test):
    classifier = skflow.TensorFlowLinearClassifier(n_classes=2)
    return classifier.fit(X_train, y_train)

def custom_dnn_model(X_train, y_train):
    def _model_fn(X, y):
        layers = skflow.ops.dnn(X, [50,50,50], keep_prob=0.5)
        return skflow.models.logistic_regression(layers, y)

    classifier = skflow.TensorFlowEstimator(model_fn=_model_fn,
                                            #keep_prob=0.5,
                                            n_classes=2,
                                            batch_size=50,
                                            learning_rate=1.5,
                                            steps=2000)
    return classifier.fit(X_train, y_train)

def dnn_model(X_train, y_train):
    classifier = skflow.TensorFlowDNNClassifier(hidden_units=[50,50,50,50],
                                                n_classes=2,
                                                batch_size=50,
                                                learning_rate=1.5,
                                                steps=2000)
    return classifier.fit(X_train, y_train)

def evaluate(labels, predictions):
    print metrics.classification_report(labels, predictions)
    print 'AUC score: %f' %  metrics.roc_auc_score(labels, predictions)
    print("Accuracy: %f" % metrics.accuracy_score(labels, predictions))

def main():
    X, y, feature_names = read_dataset('churn.csv')
    X_train, X_test, y_train, y_test = scale_and_split(X, y)
    model = dnn_model(X_train, y_train)
    predictions = model.predict(X_test)
    evaluate(y_test, predictions)

if __name__ == '__main__':
    main()

'''
Best prediction

             precision    recall  f1-score   support

        0.0       0.96      0.99      0.97       585
        1.0       0.90      0.70      0.79        82

avg / total       0.95      0.95      0.95       667

AUC score: 0.842433
Accuracy: 0.953523

'''