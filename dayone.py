# encoding: utf-8
import tensorflow as tf

# ----------------u'构建图'--------------------------
# u'产生一个1x2的矩阵'
matrix1 = tf.constant([[3., 3.]])
# u'产生一个2x1的矩阵'
matrix2 = tf.constant([[2.], [2.]])
# u'矩阵乘法'
product = tf.matmul(matrix1, matrix2)
# -----------------------------------------------
# ----------------u'在一个会话中启动图'----------------
sess = tf.Session()
result = sess.run(product)
print(result)
sess.close()
# -----------------------------------------------
# 关闭会话资源使用sess.close(),或者使用with,如下
# with tf.Session as sess:
#   result = sess.run([product])
#   print(result)

with tf.Session() as sess1:
    with tf.device("/cpu:0"):  # 指定第一个gpu执行
        # u'产生一个1x2的矩阵'
        matrix3 = tf.constant([[3., 3.]])
        # u'产生一个2x1的矩阵'
        matrix4 = tf.constant([[2.], [2.]])
        # u'矩阵乘法'
        product1 = tf.matmul(matrix3, matrix4)
        result1 = sess1.run([product1])
        print(result1)

sess3 = tf.InteractiveSession()
# 使用初始化器 initializer op 的 run() 方法初始化 'x'
x = tf.Variable([1.0, 2.0])
a = tf.constant([3.0, 3.0])
x.initializer.run()
# 增加一个减法 sub op, 从 'x' 减去 'a'. 运行减法 op, 输出结果
sub = tf.sub(x, a)
print sub.eval()

# ---------------------------------概念-------------------------------------------------------------------------------------------
# Tensor： TensorFlow 程序使用 tensor 数据结构来代表所有的数据, 计算图中, 操作间传递的数据都是 tensor.
#        你可以把 TensorFlow tensor 看作是一个 n 维的数组或列表. 一个 tensor 包含一个静态类型 rank, 和 一个 shape.
# 变量维护图执行过程中的状态信息.
# ----------------------------------------------------------------------------------------------------------------------------
state = tf.Variable(0, name="counter")  # 创建一个变量初始化位标量0
# 创建一个操作使state加1
one = tf.constant(1)
new_value = tf.add(state, one)
update = tf.assign(state, new_value)
# 启动图后, 变量必须先经过`初始化` (init) op 初始化,
# 首先必须增加一个`初始化` op 到图中.
init_op = tf.initialize_all_variables()
# 启动图, 运行 op
with tf.Session() as sess4:
    sess4.run(init_op)
    print sess4.run(state)
    for _ in range(3):
        sess4.run(update)
        print sess4.run(state)


