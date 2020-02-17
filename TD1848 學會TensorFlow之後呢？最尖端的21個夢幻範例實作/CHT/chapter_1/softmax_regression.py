# coding:utf-8
# 匯入tensorflow。
# 這句import tensorflow as tf是匯入TensorFlow約定俗成的做法，請大家記住。
import tensorflow as tf
# 匯入MNIST教學的模組
from tensorflow.examples.tutorials.mnist import input_data
# 與之前一樣，讀入MNIST資料
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# 建立x，x是一個占位符（placeholder），代表待識別的圖片
x = tf.placeholder(tf.float32, [None, 784])

# W是Softmax模型的參數，將一個784維的輸入轉為一個10維的輸出
# 在TensorFlow中，變數的參數用tf.Variable表示
W = tf.Variable(tf.zeros([784, 10]))
# b是又一個Softmax模型的參數，我們一般叫做“偏置項”（bias）。
b = tf.Variable(tf.zeros([10]))

# y=softmax(Wx + b)，y表示模型的輸出
y = tf.nn.softmax(tf.matmul(x, W) + b)

# y_是實際的圖形標簽，同樣以占位符表示。
y_ = tf.placeholder(tf.float32, [None, 10])

# 至此，我們得到了兩個重要的Tensor：y和y_。
# y是模型的輸出，y_是實際的圖形標簽，不要忘了y_是獨熱表示的
# 下面我們就會根據y和y_建構損失

# 根據y, y_建構交叉熵損失
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y)))

# 有了損失，我們就可以用隨機梯度下降針對模型的參數（W和b）進行改善
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

# 建立一個Session。只有在Session中才能執行改善步驟train_step。
sess = tf.InteractiveSession()
# 執行之前必須要起始化所有變數，分配記憶體。
tf.global_variables_initializer().run()
print('start training...')

# 進行1000步梯度下降
for _ in range(1000):
    # 在mnist.train中取100個訓練資料
    # batch_xs是形狀為(100, 784)的圖形資料，batch_ys是形如(100, 10)的實際標簽
    # batch_xs, batch_ys對應著兩個占位符x和y_
    batch_xs, batch_ys = mnist.train.next_batch(100)
    # 在Session中執行train_step，執行時要傳入占位符的值
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

# 正確的預測結果
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
# 計算預測準確率，它們都是Tensor
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# 在Session中執行Tensor可以得到Tensor的值
# 這裡是取得最終模型的正確率
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))  # 0.9185
