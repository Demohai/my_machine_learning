'''
使用tensorflow构建一个KNN算法
基本KNN实现的步骤：
1.计算训练集中的所有点与测试集中当前点的距离，距离包括L1距离和L2距离等
2.按照距离递增顺序进行排序
3.选择与当前点距离最小的k个点
4.确定k个点所在类别中数量最多的那个类别
5.返回k个点出现频率最高的类别作为当前点的类别

本例子只选取与当前测试点距离最小的训练例子，相当于将k默认为1
该例子使用MNIST数据
(http://yann.lecun.com/exdb/mnist/)

作者: 朱海
地址: https://github.com/Demohai/my_machine_learning/Basic_models
'''


import numpy as np
import tensorflow as tf

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# In this example, we limit mnist data
Xtr, Ytr = mnist.train.next_batch(5000)  # 5000 for training (nn candidates)
Xte, Yte = mnist.test.next_batch(200)  # 200 for testing

# tf Graph Input
xtr = tf.placeholder("float", [None, 784])
xte = tf.placeholder("float", [784])

# Nearest Neighbor calculation using L1 Distance
# Calculate L1 Distance
distance = tf.reduce_sum(tf.abs(xtr - xte), axis=1)  # axis=1,reduce row
# Prediction: Get min distance index from 5000 training examples(Nearest neighbor)
# The dimension of pred vector maybe 1 or more than 1,because there are maybe one training example
# or some training examples with the same distance to the current test example
pred = tf.argmin(distance, axis=0)  # axis=0,reduce column

accuracy = 0.

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Start training
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)

    # loop over test data
    for i in range(len(Xte)):
        # Get nearest neighbor
        nn_index = sess.run(pred, feed_dict={xtr: Xtr, xte: Xte[i, :]})
        # Get nearest neighbor class label and compare it to its true label
        print("Test", i, "Prediction:", np.argmax(Ytr[nn_index]), \
            "True Class:", np.argmax(Yte[i]))
        # Calculate accuracy
        if np.argmax(Ytr[nn_index]) == np.argmax(Yte[i]):
            accuracy += 1./len(Xte)
    print("Done!")
    print("Accuracy:", accuracy)
