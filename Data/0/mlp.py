from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys


from tensorflow.examples.tutorials.mnist import input_data

import scipy.io
import numpy as np
import sklearn as sk
import sklearn.utils as sku
import sklearn.preprocessing as prep
import tensorflow as tf
import data_method as dat
import copy 

FLAGS = None



def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def main(_):
    # import data from the file
    whole_data, whole_label = dat.getData()
    whole_data, whole_label = dat.shuffleData(whole_data,whole_label)
    testData = whole_data[101:129]
    data = whole_data[0:100]
    testLabel = whole_label[101:129]
    label = whole_label[0:100]

    # First layer (3 value * 19 channels)
    x =  tf.placeholder(tf.float32, [None, 57])
    w_0 = weight_variable([57, 57])

    b_0 = bias_variable([57]) 
    # Node 0
    n_0 = tf.matmul(x,w_0) + b_0


    w_1 = weight_variable([57, 57])
    
    b_1 = bias_variable([57])

    n_1 = tf.matmul(n_0,w_1) + b_1

    w_2 = weight_variable([57,3])
    b_2 = bias_variable([3])
    y = tf.matmul(n_1, w_2) + b_2

    # There are three outputs
    y_ = tf.placeholder(tf.float32, [None, 3])

    cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
    train_step = tf.train.GradientDescentOptimizer(0.0001).minimize(cross_entropy)

    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()


    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # Train
    for i in range(1000):
        batch_xs, batch_ys = dat.shuffleData(data,label)
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
        train_accuracy = accuracy.eval(feed_dict={
                x:batch_xs, y_: batch_ys})
        if i % 100 == 0:
            print("step %d, training accuracy %g"%(i, train_accuracy))


    print(sess.run(accuracy, feed_dict={x:testData,y_:testLabel}))
    # print(sess.run(y,feed_dict={x:batch_xs,y_:batch_ys}))
    # print(sess.run(y_,feed_dict={x:batch_xs,y_:batch_ys} ))



if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)


