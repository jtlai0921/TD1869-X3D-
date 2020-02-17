# coding: utf-8
from __future__ import print_function
import os
from io import BytesIO
import numpy as np
from functools import partial
import PIL.Image
import scipy.misc
import tensorflow as tf


graph = tf.Graph()
model_fn = 'tensorflow_inception_graph.pb'
sess = tf.InteractiveSession(graph=graph)
with tf.gfile.FastGFile(model_fn, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
t_input = tf.placeholder(np.float32, name='input')
imagenet_mean = 117.0
t_preprocessed = tf.expand_dims(t_input - imagenet_mean, 0)
tf.import_graph_def(graph_def, {'input': t_preprocessed})


def savearray(img_array, img_name):
    scipy.misc.toimage(img_array).save(img_name)
    print('img saved: %s' % img_name)


def render_naive(t_obj, img0, iter_n=20, step=1.0):
    # t_score是改善目的。它是t_obj的平均值
    # 結合呼叫處看，實際上就是layer_output[:, :, :, channel]的平均值
    t_score = tf.reduce_mean(t_obj)
    # 計算t_score對t_input的梯度
    t_grad = tf.gradients(t_score, t_input)[0]

    # 建立新圖
    img = img0.copy()
    for i in range(iter_n):
        # 在sess中計算梯度，以及目前的score
        g, score = sess.run([t_grad, t_score], {t_input: img})
        # 對img套用梯度。step可以看做“研讀率”
        g /= g.std() + 1e-8
        img += g * step
        print('score(mean)=%f' % (score))
    # 儲存圖片
    savearray(img, 'naive.jpg')

# 定義卷冊積層、通道數，並取出對應的tensor
name = 'mixed4d_3x3_bottleneck_pre_relu'
channel = 139
layer_output = graph.get_tensor_by_name("import/%s:0" % name)

# 定義原始的圖形噪聲
img_noise = np.random.uniform(size=(224, 224, 3)) + 100.0
# 呼叫render_naive函數著色
render_naive(layer_output[:, :, :, channel], img_noise, iter_n=20)
