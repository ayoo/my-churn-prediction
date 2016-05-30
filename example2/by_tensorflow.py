import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn import cross_validation
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn import linear_model

batch_size = 50
num_hidden_layers = 2
num_hidden_units = 4
l2_reg_param = 0.5e-3
starter_learning_rate = 0.3
num_steps = 5000
num_features = 10
num_labels = 2
dropout_rate = 0.5
test_size = 0.3

# read and transform features for machine learning friendly ones
def read_dataset(filename):
    used_cols = ['nth_membership_period','Window','Usage_Days','Supercategory_Ratio',
                 'Intensity_Ratio','Product_Article','Overview_Detail','Email_Open_R',
                 'Email_Clicked_R','Browser_Researcher', 'Target_Group']
    data = pd.read_csv(filename, sep=',', decimal='.', header=0)[used_cols]
    data.loc[data['Target_Group'] != 'Renewers', 'Target_Group'] = 'Others'
    y = data['Target_Group'] # labels
    feature_names = data.columns
    data = data.drop('Target_Group', axis=1)
    data = data.fillna(data.mean())
    X = data.as_matrix().astype(np.float32)
    y_one_hot = pd.get_dummies(y)
    y = y_one_hot.as_matrix().astype(np.float32)
    return X, y, feature_names

def scale_and_split(X, y):
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,y, test_size=test_size)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test

def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])

def add_hidden_layer(activ_input, tf_dropout_rate, layer_num):
    with tf.name_scope("hidden_layer-%d" % layer_num):
        weights = tf.Variable(tf.truncated_normal([activ_input.get_shape()[1].value, num_hidden_units]))
        biases = tf.Variable(tf.zeros([num_hidden_units]), name="biases")
        h_net = tf.matmul(activ_input, weights) + biases
        h_activ = tf.nn.relu(h_net)
        #h1_activ = tf.nn.dropout(h1_activ, tf_dropout_rate)
        h_reg = tf.nn.l2_loss(weights)
        return h_activ, h_reg

def forward_prop(tf_dataset, tf_labels, tf_dropout_rate):
    input = tf_dataset # X
    total_hidden_regs = 0
    for i in range(num_hidden_layers):
        input, reg = add_hidden_layer(input, tf_dropout_rate, i + 1)
        total_hidden_regs += reg

    with tf.name_scope("output_layer"):
        weights = tf.Variable(tf.truncated_normal([input.get_shape()[1].value, num_labels]))
        biases = tf.Variable(tf.zeros([num_labels]), name="biases")
        out_net = tf.matmul(input, weights) + biases # logits
        out_reg = tf.nn.l2_loss(weights)

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(out_net, tf_labels))
    loss = loss + l2_reg_param * (total_hidden_regs + out_reg)
    return out_net, loss

def run_ann(X_train, X_test, y_train, y_test):
    graph = tf.Graph()
    with graph.as_default():
        tf_dataset = tf.placeholder(tf.float32, shape=(None, X_train.shape[1]))
        tf_labels = tf.placeholder(tf.float32, shape=(None, num_labels))
        global_step = tf.Variable(0, trainable=False)
        tf_dropout_rate = tf.placeholder(tf.float32)
        print tf_dataset.get_shape()[1]
        logits, loss = forward_prop(tf_dataset, tf_labels, tf_dropout_rate)
        learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, 100, 0.001, staircase=True)
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
            _, l, predictions = session.run([optimizer, loss, prediction], feed_dict=feed_dict)
            if (step % 100 == 0):
                #idx = np.random.permutation(batch_size)
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
     classifier = linear_model.LogisticRegression(penalty='l2', random_state=101, class_weight='auto')

def main():
    X, y, feature_names = read_dataset('/Users/andyyoo/scikit_learn_data/renewer/Orange_Dataset.csv')
    X_train, X_test, y_train, y_test = scale_and_split(X, y)
    run_ann(X_train, X_test, y_train, y_test)

if __name__ == '__main__':
    main()