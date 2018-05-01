import tensorflow as tf

# 预处理
def prefile(filename):
    # 生成一个先入先出队列和一个Queuerunner，生成文件名队列
    filename_queue = tf.train.string_input_producer(filename, shuffle=False)
    # 定义reader
    reader = tf.TextLineReader()
    key, value = reader.read(filename_queue)
    return value


def one_reader_inorder(value, batch_size, capacity=200, num_threads=2):
    #  定义 decoder
    example, label = tf.decode_csv(value, record_defaults=[['null'], ['null']])  # ['null']解析为string类型 ，[1]为整型，[1.0]解析为浮点。
    example_batch, label_batch = tf.train.batch([example, label], batch_size=batch_size, capacity=capacity, num_threads=num_threads)  # 保证样本和标签一一对应  正常顺序
    return example_batch, label_batch

def one_reader_disorder(value, batch_size, capacity=200, num_threads=2):
    #  定义 decoder
    example, label = tf.decode_csv(value, record_defaults=[['null'], ['null']])  # ['null']解析为string类型 ，[1]为整型，[1.0]解析为浮点。
    example_batch, label_batch = tf.train.shuffle_batch([example, label], batch_size=batch_size, capacity=capacity, min_after_dequeue=100, num_threads=num_threads)  # 乱序
    return example_batch, label_batch

def multiple_readers_inorder(value, batch_size):
    # 定义了多种解码器,每个解码器跟一个reader相连
    example_list = [tf.decode_csv(value, record_defaults=[['null'], ['null']]) for _ in range(2)]  # Reader设置为2
    # 使用tf.train.batch_join()，可以使用多个reader，并行读取数据。每个Reader使用一个线程。
    example_batch, label_batch = tf.train.batch_join(example_list, batch_size=batch_size)
    return example_batch, label_batch

def multiple_readers_disorder(value, batch_size):
    # 定义了多种解码器,每个解码器跟一个reader相连
    example_list = [tf.decode_csv(value, record_defaults=[['null'], ['null']]) for _ in range(2)]  # Reader设置为2
    # 使用tf.train.batch_join()，可以使用多个reader，并行读取数据。每个Reader使用一个线程。
    example_batch, label_batch = tf.train.shuffle_batch_join(example_list, batch_size=batch_size, capacity=200, min_after_dequeue=100)
    return example_batch, label_batch


# 选择读取模式文件
def choose(i, batch_size, value):
    if i>0 and i<5:
        if i == 1:
            example_batch, label_batch = one_reader_inorder(value, batch_size, capacity=200, num_threads=2)

        elif i == 2:
            example_batch, label_batch = one_reader_disorder(value, batch_size, capacity=200, num_threads=2)

        elif i == 3:
            example_batch, label_batch = multiple_readers_inorder(value, batch_size)

        elif i == 4:
            example_batch, label_batch = multiple_readers_disorder(value, batch_size)

        return example_batch, label_batch

    else:
        print("i you input is out of range")
        exit(0)

# 线程运行文件
def run_Session(num_input, e_l_list):
    # 运行图
    with tf.Session() as sess:
        coord = tf.train.Coordinator()  # 创建一个协调器，管理线程
        threads = tf.train.start_queue_runners(coord=coord)  # 启动QueueRunner，此时文件名队列已经进队
        for i in range(num_input):
            e_val, l_val = sess.run(e_l_list)
            print(e_val, l_val)
        coord.request_stop()
        coord.join(threads)

# 主函数
def main(i, batch_size, num_input, filename=['A.csv', 'B.csv', 'C.csv']):
    value = prefile(filename)
    example_batch, label_batch = choose(i, batch_size, value)
    run_Session(num_input, [example_batch, label_batch])



main(2, 4, 9)