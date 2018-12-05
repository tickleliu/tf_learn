#-*- coding:utf-8 -*-  
""" 
@author:mlliu
@file: tf_gradient_loss.py 
@time: 2018/11/30 
"""
import argparse
import sys
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

FLAGS = None


def main(_):
    # Import data
    mnist = input_data.read_data_sets("/home/liuml/model_data/mnist", one_hot=True,
                                      validation_size=10000)

    # Create the model
    x = tf.placeholder(tf.float32, [None, 784])
    W_2 = tf.Variable(tf.random_normal([784, 1]) / tf.sqrt(784.0))
    b_2 = tf.Variable(tf.random_normal([1]))
    z_2 = tf.matmul(x, W_2) + b_2
    a_2 = tf.sigmoid(z_2)

    W_3 = tf.Variable(tf.random_normal([1, 1]) / tf.sqrt(1.0))
    b_3 = tf.Variable(tf.random_normal([1]))
    z_3 = tf.matmul(a_2, W_3) + b_3
    a_3 = tf.sigmoid(z_3)

    W_4 = tf.Variable(tf.random_normal([1, 1]) / tf.sqrt(1.0))
    b_4 = tf.Variable(tf.random_normal([1]))
    z_4 = tf.matmul(a_3, W_4) + b_4
    a_4 = tf.sigmoid(z_4)

    W_5 = tf.Variable(tf.random_normal([1, 1]) / tf.sqrt(1.0))
    b_5 = tf.Variable(tf.random_normal([1]))
    z_5 = tf.matmul(a_4, W_5) + b_5
    a_5 = tf.sigmoid(z_5)

    W_6 = tf.Variable(tf.random_normal([1, 10]) / tf.sqrt(1.0))
    b_6 = tf.Variable(tf.random_normal([10]))
    z_6 = tf.matmul(a_5, W_6) + b_6
    a_6 = tf.sigmoid(z_6)

    # Define loss and optimizer
    y_ = tf.placeholder(tf.float32, [None, 10])

    tf.add_to_collection(tf.GraphKeys.WEIGHTS, W_2)
    tf.add_to_collection(tf.GraphKeys.WEIGHTS, W_3)
    tf.add_to_collection(tf.GraphKeys.WEIGHTS, W_4)
    tf.add_to_collection(tf.GraphKeys.WEIGHTS, W_5)
    tf.add_to_collection(tf.GraphKeys.WEIGHTS, W_6)
    regularizer = tf.contrib.layers.l2_regularizer(scale=5.0 / 50000)
    reg_term = tf.contrib.layers.apply_regularization(regularizer)

    loss = (tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(labels=y_, logits=z_6)) +
        reg_term)

    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()

    correct_prediction = tf.equal(tf.argmax(a_6, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    grad_1_hidden_layer = tf.gradients(xs=[b_2], ys=loss)
    grad_2_hidden_layer = tf.gradients(xs=[b_3], ys=loss)
    grad_3_hidden_layer = tf.gradients(xs=[b_4], ys=loss)
    grad_4_hidden_layer = tf.gradients(xs=[b_5], ys=loss)

    grad_value_1 = tf.norm(grad_1_hidden_layer)
    grad_value_2 = tf.norm(grad_2_hidden_layer)
    grad_value_3 = tf.norm(grad_3_hidden_layer)
    grad_value_4 = tf.norm(grad_4_hidden_layer)

    tf.summary.scalar('grad_value_1', grad_value_1)
    tf.summary.scalar('grad_value_2', grad_value_2)
    tf.summary.scalar('grad_value_3', grad_value_3)
    tf.summary.scalar('grad_value_4', grad_value_4)

    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(
        'MNIST/logs/tf2-2/one-path', sess.graph)

    # Train
    best = 0
    for epoch in range(60):
        for _ in range(5000):
            batch_xs, batch_ys = mnist.train.next_batch(10)
            sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
        summary = sess.run(merged, feed_dict={x: batch_xs, y_: batch_ys})
        train_writer.add_summary(summary, epoch)
        # Test trained model
        accuracy_currut_train = sess.run(
            accuracy,
            feed_dict={x: mnist.train.images,
                       y_: mnist.train.labels})

        accuracy_currut_validation = sess.run(
            accuracy,
            feed_dict={x: mnist.validation.images,
                       y_: mnist.validation.labels})

        print("Epoch %s: train: %s validation: %s"
              % (epoch, accuracy_currut_train, accuracy_currut_validation))
        best = (best, accuracy_currut_validation)[
            best <= accuracy_currut_validation]

    # Test trained model
    print("best: %s" % best)
    train_writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='../MNIST/',
                        help='Directory for storing input data')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)