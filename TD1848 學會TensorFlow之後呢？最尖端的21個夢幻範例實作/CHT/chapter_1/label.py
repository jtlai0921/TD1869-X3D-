# coding: utf-8
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
# 讀取mnist資料集。若果不存在會事先下載。
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# 看前20張訓練圖片的label
for i in range(20):
    # 得到one-hot表示，形如(0, 1, 0, 0, 0, 0, 0, 0, 0, 0)
    one_hot_label = mnist.train.labels[i, :]
    # 透過np.argmax我們可以直接獲得原始的label
    # 因為只有1位為1，其他都是0
    label = np.argmax(one_hot_label)
    print('mnist_train_%d.jpg label: %d' % (i, label))
