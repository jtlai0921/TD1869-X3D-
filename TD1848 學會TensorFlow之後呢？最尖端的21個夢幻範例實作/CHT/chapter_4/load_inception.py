# coding:utf-8
# 匯入要用到的基本模組。
from __future__ import print_function
import numpy as np
import tensorflow as tf

# 建立圖和Session
graph = tf.Graph()
sess = tf.InteractiveSession(graph=graph)

# tensorflow_inception_graph.pb檔案中，既儲存了inception的網路結構也儲存了對應的資料
# 使用下面的敘述將之匯入
model_fn = 'tensorflow_inception_graph.pb'
with tf.gfile.FastGFile(model_fn, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
# 定義t_input為我們輸入的圖形
t_input = tf.placeholder(np.float32, name='input')
imagenet_mean = 117.0
# 輸入圖形需要經由處理才能送入網路中
# expand_dims是加一維，從[height, width, channel]變成[1, height, width, channel]
# t_input - imagenet_mean是減去一個均值
t_preprocessed = tf.expand_dims(t_input - imagenet_mean, 0)
tf.import_graph_def(graph_def, {'input': t_preprocessed})

# 找到所有卷冊積層
layers = [op.name for op in graph.get_operations() if op.type == 'Conv2D' and 'import/' in op.name]

# 輸出卷冊積層層數
print('Number of layers', len(layers))

# 特別地，輸出mixed4d_3x3_bottleneck_pre_relu的形狀
name = 'mixed4d_3x3_bottleneck_pre_relu'
print('shape of %s: %s' % (name, str(graph.get_tensor_by_name('import/' + name + ':0').get_shape())))
