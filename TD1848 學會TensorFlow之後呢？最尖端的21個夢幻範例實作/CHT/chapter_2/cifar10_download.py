# coding:utf-8
# 引入目前目錄中的已經撰寫好的cifar10模組
import cifar10
# 引入tensorflow
import tensorflow as tf

# tf.app.flags.FLAGS是TensorFlow內定的一個全局變數儲存器，同時可以用於指令行參數的處理
FLAGS = tf.app.flags.FLAGS
# 在cifar10模組中預先定義了f.app.flags.FLAGS.data_dir為CIFAR-10的資料路徑
# 我們把這個路徑改為cifar10_data
FLAGS.data_dir = 'cifar10_data/'

# 若果不存在資料檔，就會執行下載
cifar10.maybe_download_and_extract()
