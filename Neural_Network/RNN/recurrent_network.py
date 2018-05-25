""" 循环网络

使用tensorflow构建一个循环神经网络
该例程使用MNIST数据集(http://yann.lecun.com/exdb/mnist/)

链接:
    [Long Short Term Memory](http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf)
    [MNIST Dataset](http://yann.lecun.com/exdb/mnist/).

作者: 朱海
文件地址: https://github.com/Demohai/my_tensorflow_learn/tree/master/Neural_Network/RNN
"""

from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib import rnn

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

'''
为了使用循环神经网络区分图像，我们将每一幅图像视作像素点序列。因为MNIST图像的形状是28*28，
所以我们后面会将其转化成28个序列，即28个时间步，每个时间步的输入有28个像素。
'''

# 训练参数全局变量
learning_rate = 0.001
training_steps = 10000
batch_size = 128
display_step = 200

# 神经网络参数
num_input = 28  # 每一个序列的输入的维数 (img shape: 28*28)
timesteps = 28  # 时间步
num_hidden = 128  # 每一时间步，连接输入的隐藏层，隐藏节点的数量
num_classes = 10  # MNIST 总的分类数(0-9 digits)

# 计算图输入
X = tf.placeholder("float", [None, timesteps, num_input])
Y = tf.placeholder("float", [None, num_classes])

# 定义LSTM输出的权重变量
weights = {
    'out': tf.Variable(tf.random_normal([num_hidden, num_classes])) # weights['out']为array（数组）
}
biases = {
    'out': tf.Variable(tf.random_normal([num_classes]))
}

# 定义循环神经网络
def RNN(X, weights, biases):

    # 将输入数据的形状进行重构，满足RNN的需求
    # 输入数据的形状：（batch_size, timesteps, num_input）
    # 需要的数据形状：timesteps个形状为（batch_size, num_input）的数据

    # 将X打乱，重构成具有timesteps个(batch_size, num_input)数据的列表
    X = tf.unstack(X, timesteps, 1)

    # 用tensorflow定义一个LSTM单元
    lstm_cell = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)

    # 获得LSTM单元的输出
    outputs, states = rnn.static_rnn(lstm_cell, X, dtype=tf.float32)


    logits = tf.matmul(outputs[-1], weights['out']) + biases['out']
    return outputs, logits

outputs, logits = RNN(X, weights, biases)
prediction = tf.nn.softmax(logits)

# 定义损失和梯度优化器
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=Y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

# 评估模型
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# 初始化变量(i.e. 将它们设为默认值)
init = tf.global_variables_initializer()

# 开始训练
with tf.Session() as sess:

    # 运行初始化操作
    sess.run(init)

    for step in range(1, training_steps+1):
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        # 将batch_x转换成输入X的形状
        batch_x = batch_x.reshape((batch_size, timesteps, num_input))
        # 运行梯度优化操作（后向传播）
        sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})
        if step % display_step == 0 or step == 1:
            # 计算损失和精度  outputs_test和logits_test是用来检测RNN的返回outputs和logits的形状的
            loss, acc, outputs_test, logits_test = sess.run([loss_op, accuracy, outputs, logits], feed_dict={X: batch_x,
                                                                                                             Y: batch_y})
            # outputs_test是具有28（timesteps）个元素的list，每个元素为（128，128）的数组
            #                                                 （batch_size, num_hidden）
            print("Step " + str(step) + ", Minibatch Loss= " + \
                  "{:.4f}".format(loss) + ", Training Accuracy= " + \
                  "{:.3f}".format(acc))

    print("Optimization Finished!")

    # 计算128个测试图像的精度
    test_len = 128
    test_data = mnist.test.images[:test_len].reshape((-1, timesteps, num_input))
    test_label = mnist.test.labels[:test_len]
    print("Testing Accuracy:", \
        sess.run(accuracy, feed_dict={X: test_data, Y: test_label}))
