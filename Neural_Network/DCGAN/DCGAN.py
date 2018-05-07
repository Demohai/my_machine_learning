"""
深度卷积生成对抗网络
用tensorflow构造一个深度卷积生成对抗网络，生成手写数字图片
·作者：朱海
·项目地址：https://github.com/Demohai/my_tensorflow_learn/tree/master/Neural_Network/DCGAN
"""
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# 导入数据
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./MNIST_data_set/", one_hot=True)

# 设置训练参数
num_steps = 10000
batch_size = 32
learning_rate = 0.002

# 设置网络参数
noise_input = 100
image_input = 784

# 定义网络输入
g_noise_input = tf.placeholder(tf.float32, shape=[None, noise_input])
d_image_input = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])
is_training = tf.placeholder(tf.bool)

#leakyrelu改善传统relu容易死的问题
#当x>0，y=x; 当x<0,y=alpha * x
def leakyrelu(x, alpha=0.2):
    return 0.5 * (1 + alpha) * x + 0.5 * (1 - alpha) * abs(x)

# 生成器网络
def generator(g_input, reuse=False):
    with tf.variable_scope('Generator', reuse=reuse):
        """
        # 构建全连接层，layers根据输入自动创建变量和计算它们的形状
        x = tf.layers.dense(g_input, units=7 * 7 * 128)
        # 标准化
        x = tf.layers.batch_normalization(x, training=is_training)
        x = tf.nn.relu(x)
        # 在一层全连接层后，紧接反卷积网络，需要将x的形状(batch_size， 7*7*128)变成(batch_size, height, width, channels)形状
        # 新形状为(batch_size, 7, 7, 128)
        x = tf.reshape(x, shape=[-1, 7, 7, 128])
        # 反卷积，输入x，输出image[batch_size, 14, 14, 64]
        x = tf.layers.conv2d_transpose(x, 64, 5, strides=2, padding='same')
        x = tf.layers.batch_normalization(x, training=is_training)
        x = tf.nn.relu(x)
        # 反卷积，输入x，输出image[batch_size, 28, 28, 1]
        x = tf.layers.conv2d_transpose(x, 1, 5, strides=2, padding='same')
        # 使用tanh()激活函数，稳定性更好，使输出值在[-1, 1]之间
        g_output = tf.nn.tanh(x)

        """
        ### 尝试的另一种网络架构 ###
        # 构建全连接层，layers根据输入自动创建变量和计算它们的形状
        x = tf.layers.dense(g_input, units=6 * 6 * 128)
        x = tf.nn.tanh(x)
        # 在一层全连接层后，紧接反卷积网络，需要将x的形状(batch_size， 6*6*128)变成(batch_size, height, width, channels)形状
        # 新形状为(batch_size, 6, 6, 128)
        x = tf.reshape(x, shape=[-1, 6, 6, 128])
        # 反卷积，输入x，输出image[batch_size, 14, 14, 64]
        x = tf.layers.conv2d_transpose(x, 64, 4, strides=2)
        # 反卷积，输入x，输出image[batch_size, 28, 28, 1]
        x = tf.layers.conv2d_transpose(x, 1, 2, strides=2)
        # 使用sigmoid()激活函数，稳定性更好，使输出值在[0, 1]之间
        g_output = tf.nn.sigmoid(x)
    return g_output

# 鉴别器网络
def discriminator(d_input, reuse=False):
    with tf.variable_scope('Discriminator', reuse=reuse):
        """
        # 正向卷积网络，输入image，输出scalar
        x = tf.layers.conv2d(d_input, 64, 5, strides=2, padding='same')
        x = tf.layers.batch_normalization(x, training=is_training)
        x = leakyrelu(x)

        x = tf.layers.conv2d(x, 128, 5, strides=2, padding='same')
        x = tf.layers.batch_normalization(x, training=is_training)
        x = leakyrelu(x)
        # 全连接层，将x[batch_size, 7, 7, 128]变成[batch_size, 7*7*128]
        x = tf.reshape(x, shape=[-1, 7 * 7 * 128])
        x = tf.layers.dense(x, units=1024)
        x = tf.layers.batch_normalization(x, training=is_training)
        x = leakyrelu(x)
        # 输出层
        d_output = tf.layers.dense(x, units=2)

        """
        ### 尝试的另一种网络结构 ###
        # 正向卷积网络，输入image，输出scalar
        x = tf.layers.conv2d(d_input, 64, 5)
        x = tf.nn.tanh(x)
        x = tf.layers.average_pooling2d(x, 2, 2)
        x = tf.layers.conv2d(x, 128, 5)
        x = tf.nn.tanh(x)
        x = tf.layers.average_pooling2d(x, 2, 2)
        # 全连接层，将x[batch_size, 7, 7, 128]变成[batch_size, 7*7*128]
        x = tf.contrib.layers.flatten(x)
        x = tf.layers.dense(x, units=1024)
        x = tf.nn.tanh(x)
        # 输出层
        d_output = tf.layers.dense(x, units=2)


    return d_output


# 构建网络
def train_network(noise_input, batch_size, num_steps, learning_rate, g_noise_input, d_image_input):
    # 第一次调用Generator， reuse保持默认的False，创建Generator变量域下的变量
    g_output = generator(g_noise_input)
    # 第一次调用Discriminator， reuse保持默认的False，创建Discriminator变量域下的变量
    d_real_output = discriminator(d_image_input)

    # 以g_output作为输入，再次调用discriminator()函数时，与上面以d_image_input调用共享一套鉴别器权重参数，因此设置reuse=True
    d_fake_output = discriminator(g_output, reuse=True)

    # 计算损失函数
    d_real_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=d_real_output,
                                                                                labels=tf.ones([batch_size],
                                                                                               dtype=tf.int32)))
    d_fake_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=d_fake_output,
                                                                                labels=tf.zeros([batch_size],
                                                                                                dtype=tf.int32)))
    d_loss = d_real_loss + d_fake_loss

    g_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=d_fake_output,
                                                                           labels=tf.ones([batch_size],
                                                                                          dtype=tf.int32)))
    # 定义优化器
    g_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    d_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

    # 为G和D获取各自的训练参数
    g_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Generator')
    d_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Discriminator')

    # 创建训练
    g_train = g_optimizer.minimize(g_loss, var_list=g_vars)
    d_train = d_optimizer.minimize(d_loss, var_list=d_vars)

    # 初始化变量
    init = tf.global_variables_initializer()

    # 开始训练
    sess = tf.Session()
    sess.run(init)

    for i in range(1, num_steps + 1):
        # 准备数据
        batch_x, _ = mnist.train.next_batch(batch_size)
        batch_x = np.reshape(batch_x, newshape=[-1, 28, 28, 1])

        # 训练鉴别器D和生成器G
        z = np.random.uniform(-1, 1, size=[batch_size, noise_input])  # 随机生成噪声输入
        _, _, gl, dl = sess.run([g_train, d_train, g_loss, d_loss],
                                feed_dict={g_noise_input: z, d_image_input: batch_x, is_training: True})
        if i % 100 == 0 or i == 1:
            print("Step %i: Generator Loss: %f, Discriminator Loss: %f" % (i, gl, dl))

    # 利用生成器网络在噪声中产生图像
    f, a = plt.subplots(4, 10, figsize=(10, 4))
    for i in range(10):
        # Noise input.
        z = np.random.uniform(-1., 1., size=[4, noise_input])
        g = sess.run(g_output, feed_dict={g_noise_input: z})
        for j in range(4):
            # Generate image from noise. Extend to 3 channels for matplot figure.
            img = np.reshape(np.repeat(g[j][:, :, np.newaxis], 3, axis=2),
                             newshape=(28, 28, 3))
            a[j][i].imshow(img)

    f.show()
    plt.draw()
    plt.waitforbuttonpress()

train_network(noise_input, batch_size, num_steps, learning_rate, g_noise_input, d_image_input)
