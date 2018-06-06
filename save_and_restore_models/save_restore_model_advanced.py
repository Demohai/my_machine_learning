"""
使用tensorflow进行变量的存储和恢复
使用MNIST数据

作者: 朱海
项目地址: https://github.com/Demohai/my_tensorflow_learn/tree/master/save_and_restore_models
"""

from __future__ import print_function

# 导入MNIST数据
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

import tensorflow as tf

# 全局训练参数设置
learning_rate = 0.001
batch_size = 100
display_step = 1
model_path = "/tmp/model/"  # 模型变量保存路径 .ckpt后缀表明是checkpoint文件

# 神经网络参数
n_hidden_1 = 256
n_hidden_2 = 256
n_input = 784
n_classes = 10

# tf计算图输入
x = tf.placeholder("float", [None, n_input], name='images')
y = tf.placeholder("float", [None, n_classes], name='labels')


# 创建模型
def multilayer_perceptron(x, weights, biases):

    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)

    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)

    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer

# 初始化weights和biases
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

# 构建模型
pred = multilayer_perceptron(x, weights, biases)

# 定义损失函数和梯度优化器
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y), name='loss')
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, name="Adam_train_op").minimize(cost)

# 初始化所有变量
init = tf.global_variables_initializer()

# 创建一个saver操作，用来存储所有变量，weights和biases
saver = tf.train.Saver(max_to_keep=2)

# 运行第一个会话
print("Starting 1st session...")
with tf.Session() as sess:

    # 运行参数初始化操作
    sess.run(init)

    # 训练循环
    for epoch in range(50):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples/batch_size)
        # 遍历所有的batches
        for i in range(total_batch):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            # 运行优化器操作和损失计算操作
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_x,
                                                          y: batch_y})
            # 计算平均损失
            avg_cost += c / total_batch
        # 显示每一代的日志
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost=", \
                "{:.9f}".format(avg_cost))

        # 将模型变量存储到指定路径
        # 每迭代10次，存储一次
        if epoch % 10 == 0:
            saver.save(sess, model_path, global_step=(epoch+1))
    print("First Optimization Finished!")

    # 测试模型
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    # 计算精度
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print("Test accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))

    print("Model saved in file: %s" % model_path)

print("Starting 2nd session...")
with tf.Session() as sess:
    sess.run(init)
    new_saver = tf.train.import_meta_graph("/tmp/model/-41.meta")
    new_saver.restore(sess, tf.train.latest_checkpoint('/tmp/model/'))
    # 恢复placeholder
    graph = tf.get_default_graph()
    images = graph.get_tensor_by_name("images:0")
    labels = graph.get_tensor_by_name("labels:0")
    loss = graph.get_tensor_by_name("loss:0")
    print("Model restored!")

    # 恢复训练，从之前训练到的节点继续训练
    for epoch in range(50, 100):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples / batch_size)
        # 遍历所有的batches
        for i in range(total_batch):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            # 运行梯度优化和损失计算
            _, c = sess.run([optimizer, loss], feed_dict={images: batch_x,
                                                          labels: batch_y})
            # 计算平均损失
            avg_cost += c / total_batch
        # 每一代显示日志
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch + 1), "cost=", \
                "{:.9f}".format(avg_cost))
    print("Second Optimization Finished!")

    # 测试模型
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    # 计算精度
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print("Test accuracy:", accuracy.eval(
        {x: mnist.test.images, y: mnist.test.labels}))

