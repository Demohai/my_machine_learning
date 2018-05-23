"""
使用tensorboard进行图和相关参数的可视化
该例子使用MNIST数据集
·作者：朱海
·项目地址：https://github.com/Demohai/my_tensorflow_learn/tree/master/tensorboard_tutorials
"""
import tensorflow as tf

# 导入MNIST数据
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# 设置全局训练参数
learning_rate = 0.01
training_epochs = 25
batch_size = 100
display_step = 1
logs_path = '/tmp/tensorflow_logs/example/'

# 神经网络参数
n_hidden_1 = 256  # 第一隐藏层的节点数
n_hidden_2 = 256  # 第二隐藏层的节点数
n_input = 784  # MNIST数据输入特征数 (img shape: 28*28)
n_classes = 10  # MNIST总的分类数 (0-9 digits)

# 计算图的输入
# mnist数据图的形状 28*28=784
x = tf.placeholder(tf.float32, [None, 784], name='InputData')
# 0-9 字母识别 => 10类
y = tf.placeholder(tf.float32, [None, 10], name='LabelData')


# 创建全连接层
def multilayer_perceptron(x, weights, biases):
    # 隐藏层用relu激活
    layer_1 = tf.add(tf.matmul(x, weights['w1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    # 创建一个summary，可视化第一个隐藏层的relu激活值
    tf.summary.histogram("relu1", layer_1)
    # 隐藏层用relu激活
    layer_2 = tf.add(tf.matmul(layer_1, weights['w2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)
    # 创建第二个摘要，可视化第二个隐藏层的relu激活值
    tf.summary.histogram("relu2", layer_2)
    # 输出层
    out_layer = tf.add(tf.matmul(layer_2, weights['w3']), biases['b3'])
    return out_layer

# 采用正态分布函数初始化网络参数
def parameter_initialize(n_input, n_hidden_1, n_hidden_2, n_classes):
    weights = {
        'w1': tf.Variable(tf.random_normal([n_input, n_hidden_1]), name='W1'),
        'w2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2]), name='W2'),
        'w3': tf.Variable(tf.random_normal([n_hidden_2, n_classes]), name='W3')
    }
    biases = {
        'b1': tf.Variable(tf.random_normal([n_hidden_1]), name='b1'),
        'b2': tf.Variable(tf.random_normal([n_hidden_2]), name='b2'),
        'b3': tf.Variable(tf.random_normal([n_classes]), name='b3')
    }

    # 创建summary可视化weights和biases
    for var in tf.trainable_variables():
        tf.summary.histogram(var.name, var)
    return weights, biases

# 创建tensorboard图的名称域，将各操作封装在各名称域中，以进行模块化显示
def name_scope(x, y, weights, biases):
    with tf.name_scope('Model'):
        # 构建全连接层
        pred = multilayer_perceptron(x, weights, biases)

    with tf.name_scope('Loss'):
        # Softmax交叉熵损失函数
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
        tf.summary.scalar("loss", loss)

    with tf.name_scope('SGD'):
        # 选择梯度下降优化器进行参数优化
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        # 创建用于计算每一个变量的梯度的操作
        # tf.trainable_variables() 返回所有可训练的变量
        # tf.gradients(ys, xs) 返回ys对xs的导数的tensor列表
        grads = tf.gradients(loss, tf.trainable_variables())
        # zip(grads, tf.trainable_variables())将一个可训练变量和其梯度打包成一个元组[(w1_gradient, w1), ……]
        grads = list(zip(grads, tf.trainable_variables()))
        # 根据梯度更新所有变量的操作
        apply_grads = optimizer.apply_gradients(grads_and_vars=grads)

        # 创建summary，可视化weights和biases的梯度
        for grad, var in grads:
            tf.summary.histogram(var.name + '/gradient', grad)

    with tf.name_scope('Accuracy'):
        # 准确度
        acc = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
        acc = tf.reduce_mean(tf.cast(acc, tf.float32))
        tf.summary.scalar("accuracy", acc)
    return loss, acc, grads, apply_grads

# 创建整个模型，集合上面所有的操作，并运行
def model(x, y):
    weights, biases = parameter_initialize(n_input, n_hidden_1, n_hidden_2, n_classes)
    loss, acc, grads, apply_grads = name_scope(x, y, weights, biases)

    init = tf.global_variables_initializer()
    # 整合上面所有的summary操作
    merged_summary_op = tf.summary.merge_all()

    # 开始训练
    with tf.Session() as sess:

        # 运行变量初始化
        sess.run(init)

        # 创建将日志写到tensorboard的操作
        summary_writer = tf.summary.FileWriter(logs_path,
                                                graph=tf.get_default_graph())

        # 训练循环
        for epoch in range(training_epochs):
            avg_cost = 0.
            total_batch = int(mnist.train.num_examples/batch_size)
            # 遍历所有的batches
            for i in range(total_batch):
                batch_xs, batch_ys = mnist.train.next_batch(batch_size)
                # 运行优化器操作(backprop), 计算损失操作 (to get loss value)和summary节点

                # 注意：在定义运行操作后的返回值的名字时，不能与相应的操作同名
                # 如果同名，返回值会覆盖同名的操作，使操作不再是operation，而是tensor
                # 下次再运行该操作时，就会报错
                _, c, summary = sess.run([apply_grads, loss, merged_summary_op],
                                         feed_dict={x: batch_xs, y: batch_ys})
                # 添加每一次迭代的日志
                summary_writer.add_summary(summary, epoch * total_batch + i)
                # 计算平均损失
                avg_cost += c / total_batch
            # 显示每一次迭代的日志
            if (epoch+1) % display_step == 0:
                print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))

        print("Optimization Finished!")

        # 测试模型
        # 计算精度
        print("Accuracy:", acc.eval({x: mnist.test.images, y: mnist.test.labels}))

        print("Run the command line:\n" \
              "--> tensorboard --logdir=/tmp/tensorflow_logs " \
              "\nThen open http://0.0.0.0:6006/ into your web browser")


model(x, y)