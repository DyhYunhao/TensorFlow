# encoding: utf-8
import tensorflow as tf
import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot = True)

#实现回归模型
x = tf.placeholder("float", [None, 784])  #创建一个占位符，每一张图都是784维的向量，None表示此张量的第一个维度可以是任何长度的。
w = tf.Variable(tf.zeros([784, 10])) 
d = tf.Variable(tf.zeros([10]))#创建变量并用全为零的张量来初始化，W的维度是[784，10]，因为我们想要用784维的图片向量乘以它以得到一个10维的证据值向量，每一位对应不同数字类。b的形状是[10]

#实现模型
y = tf.nn.softmax(tf.matmul(x, w) + d)

#计算交叉熵
y_ = tf.placeholder("float", [None, 10])
cross_entropy = -tf.reduce_sum(y_ * tf.log(y)) #tf.log 计算 y 的每个元素的对数,tf.reduce_sum 计算张量的所有元素的总和
#TensorFlow用梯度下降算法（gradient descent algorithm）以0.01的学习速率最小化交叉熵。梯度下降算法（gradient descent algorithm）是一个简单的学习过程，TensorFlow只需将每个变量一点点地往使成本不断降低的方向移动。
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)
for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(1000)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
print sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})



# # 下载MNIST数据集到'MNIST_data'文件夹并解压
# mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# # 设置权重weights和偏置biases作为优化变量，初始值设为0
# weights = tf.Variable(tf.zeros([784, 10]))
# biases = tf.Variable(tf.zeros([10]))

# # 构建模型
# x = tf.placeholder("float", [None, 784])
# y = tf.nn.softmax(tf.matmul(x, weights) + biases)                                   # 模型的预测值
# y_real = tf.placeholder("float", [None, 10])                                        # 真实值

# cross_entropy = -tf.reduce_sum(y_real * tf.log(y))                                  # 预测值与真实值的交叉熵
# train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)        # 使用梯度下降优化器最小化交叉熵

# # 开始训练
# init = tf.initialize_all_variables()
# sess = tf.Session()
# sess.run(init)
# for i in range(1000):
#     batch_xs, batch_ys = mnist.train.next_batch(100)                                # 每次随机选取100个数据进行训练，即所谓的“随机梯度下降（Stochastic Gradient Descent，SGD）”
#     sess.run(train_step, feed_dict={x: batch_xs, y_real:batch_ys})                  # 正式执行train_step，用feed_dict的数据取代placeholder

#     if i % 100 == 0:
#         # 每训练10次后评估模型
#         correct_prediction = tf.equal(tf.argmax(y, 1), tf.arg_max(y_real, 1))       # 比较预测值和真实值是否一致
#         accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))             # 统计预测正确的个数，取均值得到准确率
#         print sess.run(accuracy, feed_dict={x: mnist.test.images, y_real: mnist.test.labels})
