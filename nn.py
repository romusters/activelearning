import os
import sys
import logging


def warn(*args, **kwargs):
    pass


import warnings

warnings.warn = warn
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

'''
Train the network, get the evaluation metrics, save the weights and get the probabilities from all the data
'''
def train_nn(path, nn_data, nn_id):
    import time
    t = time.time()
    import os
    try:
        logger.info("Creating directory for neural network id %i", nn_id)
        os.mkdir(path + str(nn_id))
    except:
        pass

    train_data = nn_data["trainset"]
    train_labels = nn_data["trainlabels"]
    train_ids = nn_data["trainids"]
    test_data = nn_data["testdata"]
    test_labels = nn_data["testlabels"]
    test_ids = nn_data["testids"]
    n_classes = nn_data["nclasses"]
    all_vectors = nn_data["allvectors"]
    all_ids = nn_data["allvectorsids"]

    import pandas as pd
    import tensorflow as tf
    import numpy as np
    from sklearn import metrics
    import pickle as p
    sess = tf.InteractiveSession()
    dim = 70
    x = tf.placeholder(tf.float32, shape=[None, dim], name="Input")
    y_ = tf.placeholder(tf.float32, shape=[None, n_classes], name="Output")
    # W = tf.Variable(tf.zeros([dim, n_classes]))
    # b = tf.Variable(tf.zeros([n_classes]))
    W = tf.Variable(tf.random_normal([dim, n_classes]))
    b = tf.Variable(tf.random_normal([n_classes]))

    # y = tf.matmul(x, W) + b
    hidden = tf.nn.relu(tf.add(tf.matmul(x, W), b))

    y = tf.nn.softmax(hidden)
    # cost = tf.reduce_mean(((y_ * tf.log(y)) +  ((1 - y_) * tf.log(1.0 - y))) * -1)
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))  # PREVIOUS softmax
    train_step = tf.train.AdagradOptimizer(1.0).minimize(cross_entropy)
    # train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cost)
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    sess.run(tf.global_variables_initializer())

    batch_size = 1000
    batch_count = len(train_data) / batch_size
    epochs = 8
    losses = []
    train_accs = []
    test_accs = []
    f1s = []

    logger.info("Start training neural network: %i", nn_id)
    for i in range(batch_count * epochs):
        begin = (i % batch_count) * batch_size
        end = (i % batch_count + 1) * batch_size
        batch_data = np.array(train_data[begin: end])
        batch_labels = np.array(train_labels[begin: end])

        _, loss = sess.run([train_step, cross_entropy], feed_dict={x: batch_data, y_: batch_labels})
        # _, loss = sess.run([train_step, cost], feed_dict={x: batch_data, y_: batch_labels})
        print loss
        test_acc = sess.run(accuracy, feed_dict={x: test_data, y_: test_labels})
        train_acc = sess.run(accuracy, feed_dict={x: batch_data, y_: batch_labels})

        losses.append(loss)
        train_accs.append(train_acc)
        test_accs.append(test_acc)
        prediction = tf.argmax(y, 1)
        y_pred = prediction.eval(feed_dict={x: test_data})
        gold = []
        for l in test_labels:
            label = list(l).index(1)
            gold.append(label)
        try:
            f1 = metrics.f1_score(gold, list(y_pred), average="weighted")
        except:
            logger.info("F1 score is set to 0.0 in labels with no predicted examples.")
        f1s.append(f1)

    path = path + str(nn_id) + "/"
    logger.info("Save metrics, probabilities and weights and biases in: %s", path)
    metrics = pd.DataFrame({ "loss": losses, "trainacc": train_accs, "testacc": test_accs, "f1": f1s})

    # metrics = pd.DataFrame({"loss": losses})
    metrics.to_hdf(path + "metrics.h5", "data", format="table")
    probs = y.eval(feed_dict={x: all_vectors}).tolist()
    dict = {}
    for i in range(n_classes):
        dict[i] = []

    for prob in probs:
        for idx in range(n_classes):
            dict[idx].append(prob[idx])

    probs_df = pd.DataFrame(dict)

    probs_df["id"] = all_ids

    probs_df.to_hdf(path + "probs.h5", "data", format="table")
    w_tmp = W.eval()
    b_tmp = b.eval()
    p.dump(w_tmp, open(path + "W", "w"))
    p.dump(b_tmp, open(path + "b", "w"))
    sess.close()
    logger.info("Training time took: %f seconds", time.time() - t)

    return w_tmp, b_tmp
