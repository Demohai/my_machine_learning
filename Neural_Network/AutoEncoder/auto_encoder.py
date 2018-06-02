""" Auto Encoder例子

使用tensorflow构建一个Auto Encoder网络
该例程使用MNIST数据集(http://yann.lecun.com/exdb/mnist/)

作者: 朱海
文件地址: https://github.com/Demohai/my_tensorflow_learn/tree/master/Neural_Network/RNN
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# Training parameters
learning_rate = 0.01
num_steps = 30000
batch_size = 256

display_step = 1000
examples_to_show = 10

# Network parameters
num_hidden_1 = 256
num_hidden_2 = 128
num_input = 784

# th Graph input(only images)
x = tf.placeholder("float", [None, num_input])

weights = {"encoder_h1": tf.Variable(tf.random_normal([num_input, num_hidden_1])),
           "encoder_h2": tf.Variable(tf.random_normal([num_hidden_1, num_hidden_2])),
           "decoder_h1": tf.Variable(tf.random_normal([num_hidden_2, num_hidden_1])),
           "decoder_h2": tf.Variable(tf.random_normal([num_hidden_1, num_input]))}

biases = {"encoder_b1": tf.Variable(tf.random_normal([num_hidden_1])),
          "encoder_b2": tf.Variable(tf.random_normal([num_hidden_2])),
          "decoder_b1": tf.Variable(tf.random_normal([num_hidden_1])),
          "decoder_b2": tf.Variable(tf.random_normal([num_input]))}

# build encoder
def encoder(x):
    layer_1 = tf.nn.sigmoid(tf.matmul(x, weights['encoder_h1']) + biases['encoder_b1'])

    layer_2 = tf.nn.sigmoid(tf.matmul(layer_1, weights['encoder_h2']) + biases['encoder_b2'])
    return layer_2

# build decoder
def decoder(x):
    layer_1 = tf.nn.sigmoid(tf.matmul(x, weights['decoder_h1']) + biases['decoder_b1'])
    layer_2 = tf.nn.sigmoid(tf.matmul(layer_1, weights['decoder_h2']) + biases['decoder_b2'])
    return layer_2

# construct model
encoder_out = encoder(x)
decoder_out = decoder(encoder_out)

# prediction
y_pred = decoder_out
# true labels are input data
y_true = x

# define loss and optimizer, minimize the squared error
loss = tf.reduce_mean(tf.pow(y_pred - y_true, 2))
optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(loss)

# initialize all variables
init = tf.global_variables_initializer()

# start training
with tf.Session() as sess:
    # run the initializer
    sess.run(init)

    # training
    for i in range(1, num_steps+1):
        # prepare data
        batch_x, _ = mnist.train.next_batch(batch_size)
        # run the optimizer and loss
        _, cost = sess.run([optimizer, loss], feed_dict={x: batch_x})
        # display logs
        if i % display_step == 0 or i == 1:
            print("Step %i: Minibatch Loss: %f" % (i, cost))

    # test
    # encode and decoder images from test set and visualize reconstruction
    n = 4
    canvas_orig = np.empty((28*n, 28*n))
    canvas_recon = np.empty((28*n, 28*n))
    for i in range(n):
        # Mnist test set
        batch_x, _ = mnist.test.next_batch(n)
        # encode and decode the digit image
        g = sess.run(decoder_out, feed_dict={x: batch_x})

        # display original images
        for j in range(n):
            # draw the original digits
            canvas_orig[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = g[j].reshape([28, 28])

        # display reconstructed images
        for j in range(n):
            # draw the reconstructed digits
            canvas_recon[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = g[j].reshape([28, 28])

    print("Original Images")
    plt.figure(figsize=(n, n))
    plt.imshow(canvas_orig, origin='upper', cmap='gray')


    print("Reconstructed Images")
    plt.figure(figsize=(n, n))
    plt.imshow(canvas_recon, origin='upper', cmap='gray')
    plt.show()