# coding:utf-8
# 從tensorflow.examples.tutorials.mnist引入模組。這是TensorFlow為了教學MNIST而提前編制的程式
from tensorflow.examples.tutorials.mnist import input_data
# 從MNIST_data/中讀取MNIST資料。這條敘述在資料不存在時，會自動執行下載
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# 檢視訓練資料的大小
print(mnist.train.images.shape)  # (55000, 784)
print(mnist.train.labels.shape)  # (55000, 10)

# 檢視驗證資料的大小
print(mnist.validation.images.shape)  # (5000, 784)
print(mnist.validation.labels.shape)  # (5000, 10)

# 檢視測試資料的大小
print(mnist.test.images.shape)  # (10000, 784)
print(mnist.test.labels.shape)  # (10000, 10)

# 列印出第0幅圖片的向量表示
print(mnist.train.images[0, :])

# 列印出第0幅圖片的標簽
print(mnist.train.labels[0, :])
