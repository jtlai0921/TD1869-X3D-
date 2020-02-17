﻿# coding:utf-8
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



def resize_ratio(img, ratio):
    min = img.min()
    max = img.max()
    img = (img - min) / (max - min) * 255
    img = np.float32(scipy.misc.imresize(img, ratio))
    img = img / 255 * (max - min) + min
    return img


def calc_grad_tiled(img, t_grad, tile_size=512):
    sz = tile_size
    h, w = img.shape[:2]
    sx, sy = np.random.randint(sz, size=2)
    img_shift = np.roll(np.roll(img, sx, 1), sy, 0)  # 先在行上做整體搬移，再在列上做整體搬移
    grad = np.zeros_like(img)
    for y in range(0, max(h - sz // 2, sz), sz):
        for x in range(0, max(w - sz // 2, sz), sz):
            sub = img_shift[y:y + sz, x:x + sz]
            g = sess.run(t_grad, {t_input: sub})
            grad[y:y + sz, x:x + sz] = g
    return np.roll(np.roll(grad, -sx, 1), -sy, 0)

k = np.float32([1, 4, 6, 4, 1])
k = np.outer(k, k)
k5x5 = k[:, :, None, None] / k.sum() * np.eye(3, dtype=np.float32)

# 這個函數將圖形分為低頻和高頻成分
def lap_split(img):
    with tf.name_scope('split'):
        # 做過一次卷冊積相當於一次“平順”，因此lo為低頻成分
        lo = tf.nn.conv2d(img, k5x5, [1, 2, 2, 1], 'SAME')
        # 低頻成分放縮到原始圖形一樣大小得到lo2，再用原始圖形img減去lo2，就得到高頻成分hi
        lo2 = tf.nn.conv2d_transpose(lo, k5x5 * 4, tf.shape(img), [1, 2, 2, 1])
        hi = img - lo2
    return lo, hi

# 這個函數將圖形img分成n層拉普拉斯金字塔
def lap_split_n(img, n):
    levels = []
    for i in range(n):
        # 呼叫lap_split將圖形分為低頻和高頻部分
        # 高頻部分儲存到levels中
        # 低頻部分再繼續分解
        img, hi = lap_split(img)
        levels.append(hi)
    levels.append(img)
    return levels[::-1]

# 將拉普拉斯金字塔復原到原始圖形
def lap_merge(levels):
    img = levels[0]
    for hi in levels[1:]:
        with tf.name_scope('merge'):
            img = tf.nn.conv2d_transpose(img, k5x5 * 4, tf.shape(hi), [1, 2, 2, 1]) + hi
    return img


# 對img做標准化。
def normalize_std(img, eps=1e-10):
    with tf.name_scope('normalize'):
        std = tf.sqrt(tf.reduce_mean(tf.square(img)))
        return img / tf.maximum(std, eps)

# 拉普拉斯金字塔標准化
def lap_normalize(img, scale_n=4):
    img = tf.expand_dims(img, 0)
    tlevels = lap_split_n(img, scale_n)
    # 每一層都做一次normalize_std
    tlevels = list(map(normalize_std, tlevels))
    out = lap_merge(tlevels)
    return out[0, :, :, :]


def tffunc(*argtypes):
    placeholders = list(map(tf.placeholder, argtypes))
    def wrap(f):
        out = f(*placeholders)
        def wrapper(*args, **kw):
            return out.eval(dict(zip(placeholders, args)), session=kw.get('session'))
        return wrapper
    return wrap


def render_lapnorm(t_obj, img0,
                   iter_n=10, step=1.0, octave_n=3, octave_scale=1.4, lap_n=4):
    # 同樣定義目的和梯度
    t_score = tf.reduce_mean(t_obj)
    t_grad = tf.gradients(t_score, t_input)[0]
    # 將lap_normalize轉為標準函數
    lap_norm_func = tffunc(np.float32)(partial(lap_normalize, scale_n=lap_n))

    img = img0.copy()
    for octave in range(octave_n):
        if octave > 0:
            img = resize_ratio(img, octave_scale)
        for i in range(iter_n):
            g = calc_grad_tiled(img, t_grad)
            # 唯一的區別在於我們使用lap_norm_func來標准化g！
            g = lap_norm_func(g)
            img += g * step
            print('.', end=' ')
    savearray(img, 'lapnorm.jpg')

if __name__ == '__main__':
    name = 'mixed4d_3x3_bottleneck_pre_relu'
    channel = 139
    img_noise = np.random.uniform(size=(224, 224, 3)) + 100.0
    layer_output = graph.get_tensor_by_name("import/%s:0" % name)
    render_lapnorm(layer_output[:, :, :, channel], img_noise, iter_n=20)
