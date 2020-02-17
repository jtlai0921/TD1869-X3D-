#coding: utf-8
# 匯入目前目錄的cifar10_input，這個模組負責讀入cifar10資料
import cifar10_input
# 匯入TensorFlow和其他一些可能用到的模組。
import tensorflow as tf
import os
import scipy.misc


def inputs_origin(data_dir):
  # filenames一共5個，從data_batch_1.bin到data_batch_5.bin
  # 讀入的都是訓練圖形
  filenames = [os.path.join(data_dir, 'data_batch_%d.bin' % i)
               for i in xrange(1, 6)]
  # 判斷檔案是否存在
  for f in filenames:
    if not tf.gfile.Exists(f):
      raise ValueError('Failed to find file: ' + f)
  # 將檔名的list包裝成TensorFlow中queue的形式
  filename_queue = tf.train.string_input_producer(filenames)
  # cifar10_input.read_cifar10是事先寫好的從queue中讀取檔案的函數
  # 傳回的結果read_input的屬性uint8image就是圖形的Tensor
  read_input = cifar10_input.read_cifar10(filename_queue)
  # 將圖片轉為實數形式
  reshaped_image = tf.cast(read_input.uint8image, tf.float32)
  # 傳回的reshaped_image是一張圖片的tensor
  # 我們應當這樣瞭解reshaped_image：每次使用sess.run(reshaped_image)，就會取出一張圖片
  return reshaped_image

if __name__ == '__main__':
  # 建立一個階段sess
  with tf.Session() as sess:
    # 呼叫inputs_origin。cifar10_data/cifar-10-batches-bin是我們下載的資料的資料夾位置
    reshaped_image = inputs_origin('cifar10_data/cifar-10-batches-bin')
    # 這一步start_queue_runner很重要。
    # 我們之前有filename_queue = tf.train.string_input_producer(filenames)
    # 這個queue必須透過start_queue_runners才能啟動
    # 缺少start_queue_runners程式將不能執行
    threads = tf.train.start_queue_runners(sess=sess)
    # 變數起始化
    sess.run(tf.global_variables_initializer())
    # 建立資料夾cifar10_data/raw/
    if not os.path.exists('cifar10_data/raw/'):
      os.makedirs('cifar10_data/raw/')
    # 儲存30張圖片
    for i in range(30):
      # 每次sess.run(reshaped_image)，都會取出一張圖片
      image_array = sess.run(reshaped_image)
      # 將圖片儲存
      scipy.misc.toimage(image_array).save('cifar10_data/raw/%d.jpg' % i)
