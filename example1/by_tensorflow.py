import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn import cross_validation
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn import linear_model

batch_size = 50
num_hidden_units = 50
l2_reg_param = 0.5e-3
learning_rate = 1.5 # This should be a lower number
num_steps = 2000
num_labels = 2
dropout_rate = 0.5

# read and transform features for machine learning friendly ones
def read_dataset(filename):
    drop_cols = ['State','Area Code','Phone','Churn?']
    yes_no_cols = ["Int'l Plan","VMail Plan"]
    churn_data = pd.read_csv(filename, sep=',', decimal='.', header=0)
    churn_data[yes_no_cols] = churn_data[yes_no_cols] == 'yes'
    y = churn_data['Churn?'] # labels
    churn_data = churn_data.drop(drop_cols, axis=1)
    feature_names = churn_data.columns
    X = churn_data.as_matrix().astype(np.float32)
    y_one_hot = pd.get_dummies(y)
    #print y_one_hot
    y = y_one_hot.as_matrix().astype(np.float32)
    return X, y, feature_names

def scale_and_split(X, y):
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,y, test_size=0.2)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test

def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])

def forward_prop(tf_dataset, tf_labels, tf_dropout_rate):
    with tf.name_scope("hidden_layer1"):
        weights = tf.Variable(tf.truncated_normal([17, num_hidden_units]))
        biases = tf.Variable(tf.zeros([num_hidden_units]), name="biases")
        h1_net = tf.matmul(tf_dataset, weights) + biases
        h1_activ = tf.nn.relu(h1_net)
        #h1_activ = tf.nn.dropout(h1_activ, tf_dropout_rate)
        h1_reg = tf.nn.l2_loss(weights)

    '''
    # Need more data for this deep network to work well.
    with tf.name_scope("hidden_layer2"):
        weights = tf.Variable(tf.truncated_normal([num_hidden_units, num_hidden_units]))
        biases = tf.Variable(tf.zeros([num_hidden_units]), name="biases")
        h2_net = tf.matmul(h1_activ, weights) + biases
        h2_activ = tf.nn.relu(h2_net)
        #h1_activ = tf.nn.dropout(h1_activ, tf_dropout_rate)
        h2_reg = tf.nn.l2_loss(weights)
    '''

    with tf.name_scope("output_layer"):
        weights = tf.Variable(tf.truncated_normal([num_hidden_units, num_labels]))
        biases = tf.Variable(tf.zeros([num_labels]), name="biases")
        out_net = tf.matmul(h1_activ, weights) + biases # logits
        out_reg = tf.nn.l2_loss(weights)

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(out_net, tf_labels))
    loss = loss + l2_reg_param * (h1_reg + out_reg)
    return out_net, loss

def run_ann(X_train, X_test, y_train, y_test):
    graph = tf.Graph()
    with graph.as_default():
        tf_dataset = tf.placeholder(tf.float32, shape=(None, X_train.shape[1]))
        tf_labels = tf.placeholder(tf.float32, shape=(None, num_labels))
        tf_dropout_rate = tf.placeholder(tf.float32)
        print tf_dataset.get_shape()[1]
        logits, loss = forward_prop(tf_dataset, tf_labels, tf_dropout_rate)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
        prediction = tf.nn.softmax(logits)

    with tf.Session(graph=graph) as session:
        tf.initialize_all_variables().run()
        print("Initialized")
        for step in range(num_steps):
            offset = (step * batch_size) % (y_train.shape[0] - batch_size)
            # Generate a minibatch.
            batch_data = X_train[offset:(offset + batch_size), :]
            batch_labels = y_train[offset:(offset + batch_size), :]
            feed_dict = {tf_dataset : batch_data, tf_labels : batch_labels, tf_dropout_rate: dropout_rate}
            _, l, predictions = session.run(
            [optimizer, loss, prediction], feed_dict=feed_dict)
            if (step % 100 == 0):
                idx = np.random.permutation(batch_size)
                print("Minibatch loss at step %d: %f" % (step, l))
                print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))

        #idx = np.random.permutation(batch_size)
        test_pred = session.run(prediction, feed_dict={tf_dataset: X_test, tf_dropout_rate: dropout_rate})
        print("\n\nTest accuracy: %.1f%%" % accuracy(test_pred, y_test))
        #correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

        y_hat = tf.argmax(test_pred,1).eval()
        y_true = tf.argmax(y_test,1).eval()

        print 'Y hat\n',y_hat
        print 'Y True\n',y_true

        print metrics.classification_report(y_true, y_hat)
        print 'AUC score: ', metrics.roc_auc_score(y_true, y_hat)
        print("Accuracy: %f" % metrics.accuracy_score(y_true, y_hat))

def run_lr():
     classifier = linear_model.LogisticRegression(penalty='l1', random_state=101, class_weight='auto')

def main():
    X, y, feature_names = read_dataset('churn.csv')
    X_train, X_test, y_train, y_test = scale_and_split(X, y)
    run_ann(X_train, X_test, y_train, y_test)

if __name__ == '__main__':
    main()


'''
Best Predictions
             precision    recall  f1-score   support

          0       0.96      0.97      0.97       578
          1       0.78      0.76      0.77        89

avg / total       0.94      0.94      0.94       667

AUC score:  0.865586485751
Accuracy: 0.940030

'''