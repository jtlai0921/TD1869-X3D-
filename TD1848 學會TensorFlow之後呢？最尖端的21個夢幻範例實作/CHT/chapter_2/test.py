# coding:utf-8
import os
if not os.path.exists('read'):
    os.makedirs('read/')

# 匯入TensorFlow
import tensorflow as tf 

# 新增一個Session
with tf.Session() as sess:
    # 我們要讀三幅圖片A.jpg, B.jpg, C.jpg
    filename = ['A.jpg', 'B.jpg', 'C.jpg']
    # string_input_producer會產生一個檔名佇列
    filename_queue = tf.train.string_input_producer(filename, shuffle=False, num_epochs=5)
    # reader從檔名佇列中讀資料。對應的方法是reader.read
    reader = tf.WholeFileReader()
    key, value = reader.read(filename_queue)
    # tf.train.string_input_producer定義了一個epoch變數，要對它進行起始化
    tf.local_variables_initializer().run()
    # 使用start_queue_runners之後，才會開始填充佇列
    threads = tf.train.start_queue_runners(sess=sess)
    i = 0
    while True:
        i += 1
        # 取得圖片資料並儲存
        image_data = sess.run(value)
        with open('read/test_%d.jpg' % i, 'wb') as f:
            f.write(image_data)
# 程式最後會拋出一個OutOfRangeError，這是epoch跑完，佇列關閉的標志
