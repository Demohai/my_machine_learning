import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/MNIST_data_set/", one_hot=True)   # 加载MNIST数据


# 定义输入数据
def input_data(Input_node, Output_node):
    x = tf.placeholder(tf.float32, shape=[None,Input_node], name='input_x')
    y = tf.placeholder(tf.float32, shape=[None,Output_node], name='output_y')
    return x, y


# 参数初始化
def parameter_initialize(Input_node, Layer1_node, Output_node):
    W1 = tf.Variable(tf.truncated_normal([Input_node, Layer1_node], stddev=0.1))
    b1 = tf.Variable(tf.constant(0.1, shape=[Layer1_node]))
    W2 = tf.Variable(tf.truncated_normal([Layer1_node, Output_node], stddev=0.1))
    b2 = tf.Variable(tf.constant(0.1, shape=[Output_node]))
    parameter = {"W1": W1, "b1": b1,
                 "W2": W2, "b2": b2}
    return parameter


# 前向过程
def inference(input_tensor, parameter):
    W1 = parameter["W1"]
    b1 = parameter["b1"]
    W2 = parameter["W2"]
    b2 = parameter["b2"]
    layer1 = tf.nn.relu(tf.matmul(input_tensor, W1) + b1)
    return tf.nn.relu(tf.matmul(layer1, W2) + b2)


# 定义损失函数,加L2正则化
def loss(y_pred, y, parameter):
    W1 = parameter["W1"]
    W2 = parameter["W2"]
    cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y_pred, labels=tf.argmax(y,1)))
    # 定义L2正则项
    regularizer = tf.contrib.layers.l2_regularizer(regularization_rate)
    regularition = regularizer(W1) + regularizer(W2)
    # 交叉熵与L2正则项相加，作为损失函数
    cost = cross_entropy + regularition
    return cost


# 训练函数，学习率衰减
def train(Learning_rate_base, Learning_rate_decay, Batch_size, cost):
    # 学习率指数衰减函数
    # tf.train.exponential_decay(learning_rate_base,  global_step,  decay_steps,
    # learning_rate_decay,  staircase=False or True,  name(optional))
    # decayed_learning_rate = learning_rate_base *learning_rate_decay^ (global_step/decay_steps)
    # 如果staircase=True,global_step/decay_steps为整除；如果staircase=True,global_step/decay_steps为正常除
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(Learning_rate_base, global_step, mnist.train.num_examples / Batch_size, learning_rate_decay, staircase=True)
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost,global_step=global_step)
    return train_step


# 定义评估函数，此程序以准确度作为评估指标
def evaluate(y, y_pred):
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_pred,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return accuracy


# 整个模型函数
def model(Input_node, Layer1_model, Output_node, Learning_rate_base, Learning_rate_decay, Batch_size):
    x, y = input_data(Input_node, Output_node)
    parameter = parameter_initialize(Input_node, Layer1_node, Output_node)
    y_pred = inference(x, parameter)
    cost = loss(y_pred, y, parameter)
    train_step = train(Learning_rate_base, Learning_rate_decay, Batch_size, cost)
    accuracy = evaluate(y, y_pred)
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        # 验证集feed
        validate_feed = {x: mnist.validation.images, y: mnist.validation.labels}
        # 测试集feed
        test_feed = {x: mnist.test.images, y: mnist.test.labels}

        for i in range(training_steps):
            if i % 1000 == 0:
                validate_accuracy = sess.run(accuracy, feed_dict=validate_feed)
                print("After %d training steps ,validation accuracy is %g" % (i, validate_accuracy))

            # 每一次遍历训练集，随机取Batch—size个数据训练
            x_batch, y_batch = mnist.train.next_batch(Batch_size)
            sess.run(train_step, feed_dict={x: x_batch, y: y_batch})

        test_accuracy = sess.run(accuracy, feed_dict=test_feed)
        print("After %d training steps ,test accuracy is %g" % (training_steps, test_accuracy))


# 定义输入层，隐藏层和输出层的节点数量
Input_node = 784
Layer1_node = 500
Output_node = 10
# 定义每一训练批次的样本数量
Batch_size = 100
# 模型相关参数
learning_rate_base = 0.1      # 基础学习率
learning_rate_decay = 0.99  # 学习率衰减因子
regularization_rate = 0.001  # L2正则化因子
training_steps = 10000        # 遍历整个训练集的次数

model(Input_node, Layer1_node, Output_node, learning_rate_base, learning_rate_decay, Batch_size)