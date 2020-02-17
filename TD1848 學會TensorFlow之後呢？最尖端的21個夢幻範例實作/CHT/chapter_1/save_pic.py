#coding: utf-8
from tensorflow.examples.tutorials.mnist import input_data
import scipy.misc
import os

# 讀取MNIST資料集。若果不存在會事先下載。
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# 我們把原始圖片儲存在MNIST_data/raw/資料夾下
# 若果沒有這個資料夾會自動建立
save_dir = 'MNIST_data/raw/'
if os.path.exists(save_dir) is False:
    os.makedirs(save_dir)

# 儲存前20張圖片
for i in range(20):
    # 請注意，mnist.train.images[i, :]就表示第i張圖片（序號從0開始）
    image_array = mnist.train.images[i, :]
    # TensorFlow中的MNIST圖片是一個784維的向量，我們重新把它復原為28x28維的圖形。
    image_array = image_array.reshape(28, 28)
    # 儲存檔案的格式為 mnist_train_0.jpg, mnist_train_1.jpg, ... ,mnist_train_19.jpg
    filename = save_dir + 'mnist_train_%d.jpg' % i
    # 將image_array儲存為圖片
    # 先用scipy.misc.toimage轉為圖形，再呼叫save直接儲存。
    scipy.misc.toimage(image_array, cmin=0.0, cmax=1.0).save(filename)

print('Please check: %s ' % save_dir)

