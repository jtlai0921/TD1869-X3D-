# coding: utf-8
from __future__ import print_function

import tensorflow as tf
import random
import numpy as np

# 這個類別用於產生序列樣本
class ToySequenceData(object):
    """ 產生序列資料。每個數量可能具有不同的長度。
    一共產生下面兩類別資料
    - 類別別 0: 線性序列 (如 [0, 1, 2, 3,...])
    - 類別別 1: 完全隨機的序列 (i.e. [1, 3, 10, 7,...])
    注意:
    max_seq_len是最大的序列長度。對於長度小於這個數值的序列，我們將會補0。
    在送入RNN計算時，會借助sequence_length這個屬性來進行對應長度的計算。
    """
    def __init__(self, n_samples=1000, max_seq_len=20, min_seq_len=3,
                 max_value=1000):
        self.data = []
        self.labels = []
        self.seqlen = []
        for i in range(n_samples):
            # 序列的長度是隨機的，在min_seq_len和max_seq_len之間。
            len = random.randint(min_seq_len, max_seq_len)
            # self.seqlen用於儲存所有的序列。
            self.seqlen.append(len)
            # 以50%的機率，隨機加入一個線性或隨機的訓練
            if random.random() < .5:
                # 產生一個線性序列
                rand_start = random.randint(0, max_value - len)
                s = [[float(i)/max_value] for i in
                     range(rand_start, rand_start + len)]
                # 長度不足max_seq_len的需要補0
                s += [[0.] for i in range(max_seq_len - len)]
                self.data.append(s)
                # 線性序列的label是[1, 0]（因為我們一共只有兩類別）
                self.labels.append([1., 0.])
            else:
                # 產生一個隨機序列
                s = [[float(random.randint(0, max_value))/max_value]
                     for i in range(len)]
                # 長度不足max_seq_len的需要補0
                s += [[0.] for i in range(max_seq_len - len)]
                self.data.append(s)
                self.labels.append([0., 1.])
        self.batch_id = 0

    def next(self, batch_size):
        """
        產生batch_size的樣本。
        若果使用完了所有樣本，會重新從頭開始。
        """
        if self.batch_id == len(self.data):
            self.batch_id = 0
        batch_data = (self.data[self.batch_id:min(self.batch_id +
                                                  batch_size, len(self.data))])
        batch_labels = (self.labels[self.batch_id:min(self.batch_id +
                                                  batch_size, len(self.data))])
        batch_seqlen = (self.seqlen[self.batch_id:min(self.batch_id +
                                                  batch_size, len(self.data))])
        self.batch_id = min(self.batch_id + batch_size, len(self.data))
        return batch_data, batch_labels, batch_seqlen



# 這一部分只是測試一下如何使用上面定義的ToySequenceData
tmp = ToySequenceData()

# 產生樣本
batch_data, batch_labels, batch_seqlen = tmp.next(32)

# batch_data是序列資料，它是一個嵌套的list，形狀為(batch_size, max_seq_len, 1)
print(np.array(batch_data).shape)  # (32, 20, 1)

# 我們之前呼叫tmp.next(32)，因此一共有32個序列
# 我們可以打出第一個序列
print(batch_data[0])

# batch_labels是label，它也是一個嵌套的list，形狀為(batch_size, 2)
# (batch_size, 2)中的“2”表示為兩類別分類別
print(np.array(batch_labels).shape)  # (32, 2)

# 我們可以打出第一個序列的label
print(batch_labels[0])

# batch_seqlen一個長度為batch_size的list，表示每個序列的實際長度
print(np.array(batch_seqlen).shape)  # (32,)

# 我們可以打出第一個序列的長度
print(batch_seqlen[0])




# 執行的參數
learning_rate = 0.01
training_iters = 1000000
batch_size = 128
display_step = 10

# 網路定義時的參數
seq_max_len = 20 # 最大的序列長度
n_hidden = 64 # 隱層的size
n_classes = 2 # 類別別數

trainset = ToySequenceData(n_samples=1000, max_seq_len=seq_max_len)
testset = ToySequenceData(n_samples=500, max_seq_len=seq_max_len)

# x為輸入，y為輸出
# None的位置實際為batch_size
x = tf.placeholder("float", [None, seq_max_len, 1])
y = tf.placeholder("float", [None, n_classes])
# 這個placeholder儲存了輸入的x中，每個序列的實際長度
seqlen = tf.placeholder(tf.int32, [None])

# weights和bias在輸出時會用到
weights = {
    'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))
}
biases = {
    'out': tf.Variable(tf.random_normal([n_classes]))
}

def dynamicRNN(x, seqlen, weights, biases):

    # 輸入x的形狀： (batch_size, max_seq_len, n_input)
    # 輸入seqlen的形狀：(batch_size, )
    

    # 定義一個lstm_cell，隱層的大小為n_hidden（之前的參數）
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden)

    # 使用tf.nn.dynamic_rnn展開時間維度
    # 此外sequence_length=seqlen也很重要，它告訴TensorFlow每一個序列應該執行多少步
    outputs, states = tf.nn.dynamic_rnn(lstm_cell, x, dtype=tf.float32,
                                sequence_length=seqlen)
    
    # outputs的形狀為(batch_size, max_seq_len, n_hidden)
    # 若果有疑問可以參考上一章內容

    # 我們希望的是取出與序列長度相對應的輸出。如一個序列長度為10，我們就應該取出第10個輸出
    # 但是TensorFlow不支援直接對outputs進行索引，因此我們用下面的方法來做：

    batch_size = tf.shape(outputs)[0]
    # 得到每一個序列真正的index
    index = tf.range(0, batch_size) * seq_max_len + (seqlen - 1)
    outputs = tf.gather(tf.reshape(outputs, [-1, n_hidden]), index)

    # 給最後的輸出
    return tf.matmul(outputs, weights['out']) + biases['out']

# 這裡的pred是logits而不是機率
pred = dynamicRNN(x, seqlen, weights, biases)

# 因為pred是logits，因此用tf.nn.softmax_cross_entropy_with_logits來定義損失
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

# 分類別準確率
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# 起始化
init = tf.global_variables_initializer()

# 訓練
with tf.Session() as sess:
    sess.run(init)
    step = 1
    while step * batch_size < training_iters:
        batch_x, batch_y, batch_seqlen = trainset.next(batch_size)
        # 每run一次就會更新一次參數
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y,
                                       seqlen: batch_seqlen})
        if step % display_step == 0:
            # 在這個batch內計算準確度
            acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y,
                                                seqlen: batch_seqlen})
            # 在這個batch內計算損失
            loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y,
                                             seqlen: batch_seqlen})
            print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))
        step += 1
    print("Optimization Finished!")

    # 最終，我們在測試集上計算一次準確度
    test_data = testset.data
    test_label = testset.labels
    test_seqlen = testset.seqlen
    print("Testing Accuracy:", \
        sess.run(accuracy, feed_dict={x: test_data, y: test_label,
                                      seqlen: test_seqlen}))
