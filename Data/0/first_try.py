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
import tensorflow as tf

FLAGS = None

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)


def randomData():
    base = scipy.io.loadmat('base_ready_data.mat')['base_ready_data']
    left = scipy.io.loadmat('left_ready_data.mat')['left_ready_data']
    right = scipy.io.loadmat('right_ready_data.mat')['right_ready_data']

    # concatenate all of them
    whole_data = np.concatenate((base,right,left),axis=2)
    # reshape it and then transpose 
    whole_data = np.reshape(whole_data,(57,128*3)).transpose()

    # Make the correct answers
    base_label = np.concatenate((np.ones(128),np.zeros(256)))
    left_label = np.concatenate((np.zeros(128), np.ones(128),np.zeros(128)))
    right_label = np.concatenate((np.zeros(256),np.ones(128)))

    whole_label = np.concatenate((base_label,left_label,right_label))
    whole_label = np.reshape(whole_label,(3,128*3)).transpose()

    random_data,random_label = sku.shuffle(whole_data,whole_label)

    return whole_data,whole_label

def main(_):
    # import data from the file
    data, label = randomData()

    # First layer (3 value * 19 channels)
    x = tf.placeholder(tf.float32, [None, 57])
    W = tf.Variable(tf.zeros([57, 3]))
    b = tf.Variable(tf.zeros([3]))
    y = tf.matmul(x, W) + b

    # There are three outputs
    y_ = tf.placeholder(tf.float32, [None, 3])

    cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()
    # Train
    for _ in range(100):
        batch_xs, batch_ys = randomData()
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    batch_xs, batch_ys = randomData()

    print(sess.run(accuracy, feed_dict={x:batch_xs,y_:batch_ys}))



if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)


