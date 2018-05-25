""" 在tensorflow中构建一个图像数据集

在这个例子中，你需要自己制作图像数据集(JPEG).
我们将会展示两种方法来构建数据集:

- 建立一个根目录，里面有两个包含图像数据的子目录
    ```
    ROOT_FOLDER
       |-------- SUBFOLDER (CLASS 0)
       |             |
       |             | ----- image1.jpg
       |             | ----- image2.jpg
       |             | ----- etc...
       |             
       |-------- SUBFOLDER (CLASS 1)
       |             |
       |             | ----- image1.jpg
       |             | ----- image2.jpg
       |             | ----- etc...
    ```

- 将所有图像数据放在一个目录下，并列出他们各自的class_ID
    ```
    ROOT_FOLDER
    |----------/path/to/image/1.jpg CLASS_ID
    |----------/path/to/image/2.jpg CLASS_ID
    |----------/path/to/image/3.jpg CLASS_ID
    |----------/path/to/image/4.jpg CLASS_ID
    |----------etc...
    ```

下面会有一些参数需要你自己来修改(标记为'修改'),
例如数据集路径。

作者: 朱海
地址: https://github.com/Demohai/my_machine_learning/data_management
"""
from __future__ import print_function

import tensorflow as tf
import os

# 数据集参数 - 修改
mode = 'folder'  # 或者'file', 如果你选择将图像数据平行放在一个根目录下
dataset_path = '/path/to/dataset/'  # file模式下的数据集文件或folder模式下的根目录

# 图像参数
N_CLASSES = 2  # 修改, class的总数
IMG_HEIGHT = 64  # 修改, 将原图像的高重新定义的大小
IMG_WIDTH = 64  # 修改, 将原图像的宽重新定义的大小
CHANNELS = 3  # 修改，数据通道，彩色图像设为3，灰度图像设为1


# 读取数据集函数
# 2种模式: 'file' or 'folder'
def read_images(dataset_path, mode, batch_size):
    # imagepaths保存的是每一幅图像的完整路径， labels中保存的是class ID
    imagepaths, labels = list(), list()
    if mode == 'file':
        # 读取数据文件
        # open() 打开文件名为dataset_path的文件，设置为只读模式
        # .read() 返回整个文件,读到文件结尾时，返回""空字符
        # .splitlines()，以文件中的换行符进行分割，返回数据列表
        data = open(dataset_path, 'r').read().splitlines()
        for d in data:
            # 获取图像文件的文件路径，并添加到imagepaths中
            imagepaths.append(d.split(' ')[0])
            # 获取图像文件的ID，并添加到labels中
            labels.append(int(d.split(' ')[1]))
    elif mode == 'folder':
        # 一个通过首字母顺序影响所有子文件夹的ID
        label = 0
        # 列出目录
        # -------------------------------------------------------------------------------------------
        # os.walk(dataset_path),遍历dataset_path目录下的文件，返回一个三元组(root, dir, files)
        # root就是dataset_path，dir是dataset_path目录下的所有子目录名，files是dataset_path目录下的所有文件名
        # -------------------------------------------------------------------------------------------

        classes = sorted(os.walk(dataset_path))
        for root, dirs, files in classes:
            for name in dirs:
                c_dir = os.path.join(dataset.path, name)
                for c_dir_root, c_dir_dirs, c_dir_files in sorted(os.walk(c_dir)):
                    # 仅保留.jpg或.jpeg文件
                    if c_dir_files.endswith('.jpg') or c_dir_files.endswith('.jpeg'):
                        imagepaths.append(os.path.join(c_dir, c_dir_files))
                        labels.append(label)
            label += 1
    else:
        raise Exception("Unknown mode.")

    # 转换成tensor
    imagepaths = tf.convert_to_tensor(imagepaths, dtype=tf.string)
    labels = tf.convert_to_tensor(labels, dtype=tf.int32)
    # 构建一个tf队列，打乱数据
    image, label = tf.train.slice_input_producer([imagepaths, labels],
                                                 shuffle=True)

    # 从磁盘读取图像
    image = tf.read_file(image)
    image = tf.image.decode_jpeg(image, channels=CHANNELS)

    # 将图像重新构建成常用的尺寸大小
    image = tf.image.resize_images(image, [IMG_HEIGHT, IMG_WIDTH])

    # 标准化
    image = image * 1.0/127.5 - 1.0

    # 创建batches
    X, Y = tf.train.batch([image, label], batch_size=batch_size,
                          capacity=batch_size * 8,
                          num_threads=4)

    return X, Y

# -----------------------------------------------
#下面是一个标准的CNN
# -----------------------------------------------
# Note that a few elements have changed (usage of queues).

# 参数
learning_rate = 0.001
num_steps = 10000
batch_size = 128
display_step = 100

# 网络参数
dropout = 0.75

# 构建输入数据
X, Y = read_images(dataset_path, mode, batch_size)


# 创建模型
def conv_net(x, n_classes, dropout, reuse, is_training):
    # 定义一个变量域，用于重用变量
    with tf.variable_scope('ConvNet', reuse=reuse):

        # 有32个滤波器，每个滤波器是5*5大小
        conv1 = tf.layers.conv2d(x, 32, 5, activation=tf.nn.relu)
        conv1 = tf.layers.max_pooling2d(conv1, 2, 2)
        conv2 = tf.layers.conv2d(conv1, 64, 3, activation=tf.nn.relu)
        conv2 = tf.layers.max_pooling2d(conv2, 2, 2)

        # 将数据展成以为向量，作为全连接层的输入
        fc1 = tf.contrib.layers.flatten(conv2)

        # 全连接层
        fc1 = tf.layers.dense(fc1, 1024)
        # 使用Dropout(如果is_training是False, 不使用dropout)
        fc1 = tf.layers.dropout(fc1, rate=dropout, training=is_training)

        # 输出层，分类
        out = tf.layers.dense(fc1, n_classes)

        out = tf.nn.softmax(out) if not is_training else out

    return out

# 因为Dropout在训练和预测时会有不同的行为，所以我们创建两个图，并分享变量

# 创建训练图
logits_train = conv_net(X, N_CLASSES, dropout, reuse=False, is_training=True)
# 创建测试图
logits_test = conv_net(X, N_CLASSES, dropout, reuse=True, is_training=False)

# 定义损失和梯度优化器
loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
    logits=logits_train, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

# 评估模型
correct_pred = tf.equal(tf.argmax(logits_test, 1), tf.cast(Y, tf.int64))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# 初始化变量
init = tf.global_variables_initializer()

# 创建存储器
saver = tf.train.Saver()

# 开始训练
with tf.Session() as sess:

    # 运行初始化
    sess.run(init)

    # 开始数据队列
    tf.train.start_queue_runners()

    # 训练循环
    for step in range(1, num_steps+1):

        if step % display_step == 0:
            _, loss, acc = sess.run([train_op, loss_op, accuracy])
            print("Step " + str(step) + ", Minibatch Loss= " + \
                  "{:.4f}".format(loss) + ", Training Accuracy= " + \
                  "{:.3f}".format(acc))
        else:
            sess.run(train_op)

    print("Optimization Finished!")

    # 保存模型
    saver.save(sess, 'my_tf_model')
