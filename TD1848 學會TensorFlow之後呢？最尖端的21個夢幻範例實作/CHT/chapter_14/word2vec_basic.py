# coding: utf-8

# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Basic word2vec example."""

# 匯入一些需要的庫
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import math
import os
import random
import zipfile

import numpy as np
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf


# 第一步: 在下面這個位址下載語料庫
url = 'http://mattmahoney.net/dc/'


def maybe_download(filename, expected_bytes):
  """
  這個函數的功能是：
      若果filename不存在，就在上面的位址下載它。
      若果filename存在，就略過下載。
      最終會檢查文字的位元組數是否和expected_bytes相同。
  """
  if not os.path.exists(filename):
    print('start downloading...')
    filename, _ = urllib.request.urlretrieve(url + filename, filename)
  statinfo = os.stat(filename)
  if statinfo.st_size == expected_bytes:
    print('Found and verified', filename)
  else:
    print(statinfo.st_size)
    raise Exception(
        'Failed to verify ' + filename + '. Can you get to it with a browser?')
  return filename

# 下載語料庫text8.zip並驗證下載
filename = maybe_download('text8.zip', 31344016)



# 將語料庫解壓，並轉換成一個word的list
def read_data(filename):
  """
  這個函數的功能是：
      將下載好的zip檔案解壓並讀取為word的list
  """
  with zipfile.ZipFile(filename) as f:
    data = tf.compat.as_str(f.read(f.namelist()[0])).split()
  return data

vocabulary = read_data(filename)
print('Data size', len(vocabulary)) # 總長度為1700萬左右
# 輸出前100個詞。
print(vocabulary[0:100])



# 第二步: 製作一個詞表，將不常見的詞變成一個UNK標誌符
# 詞表的大小為5萬（即我們只考慮最常出現的5萬個詞）
vocabulary_size = 50000


def build_dataset(words, n_words):
  """
  函數功能：將原始的單字表示變成index
  """
  count = [['UNK', -1]]
  count.extend(collections.Counter(words).most_common(n_words - 1))
  dictionary = dict()
  for word, _ in count:
    dictionary[word] = len(dictionary)
  data = list()
  unk_count = 0
  for word in words:
    if word in dictionary:
      index = dictionary[word]
    else:
      index = 0  # UNK的index為0
      unk_count += 1
    data.append(index)
  count[0][1] = unk_count
  reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
  return data, count, dictionary, reversed_dictionary

data, count, dictionary, reverse_dictionary = build_dataset(vocabulary,
                                                            vocabulary_size)
del vocabulary  # 移除已節省記憶體
# 輸出最常出現的5個單字
print('Most common words (+UNK)', count[:5])
# 輸出轉換後的資料庫data，和原來的單字（前10個）
print('Sample data', data[:10], [reverse_dictionary[i] for i in data[:10]])
# 我們下面就使用data來製作訓練集
data_index = 0



# 第三步：定義一個函數，用於產生skip-gram模型用的batch
def generate_batch(batch_size, num_skips, skip_window):
  # data_index相當於一個指標，起始為0
  # 每次產生一個batch，data_index就會對應地往後推
  global data_index
  assert batch_size % num_skips == 0
  assert num_skips <= 2 * skip_window
  batch = np.ndarray(shape=(batch_size), dtype=np.int32)
  labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
  span = 2 * skip_window + 1  # [ skip_window target skip_window ]
  buffer = collections.deque(maxlen=span)
  # data_index是目前資料開始的位置
  # 產生batch後就往後推1位（產生batch）
  for _ in range(span):
    buffer.append(data[data_index])
    data_index = (data_index + 1) % len(data)
  for i in range(batch_size // num_skips):
    # 利用buffer產生batch
    # buffer是一個長度為 2 * skip_window + 1長度的word list
    # 一個buffer產生num_skips個數的樣本
#     print([reverse_dictionary[i] for i in buffer])
    target = skip_window  # target label at the center of the buffer
#     targets_to_avoid確保樣本不重復
    targets_to_avoid = [skip_window]
    for j in range(num_skips):
      while target in targets_to_avoid:
        target = random.randint(0, span - 1)
      targets_to_avoid.append(target)
      batch[i * num_skips + j] = buffer[skip_window]
      labels[i * num_skips + j, 0] = buffer[target]
    buffer.append(data[data_index])
    # 每利用buffer產生num_skips個樣本，data_index就向後推進一位
    data_index = (data_index + 1) % len(data)
  data_index = (data_index + len(data) - span) % len(data)
  return batch, labels

# 預設情況下skip_window=1, num_skips=2
# 此時就是從連續的3(3 = skip_window*2 + 1)個詞中產生2(num_skips)個樣本。
# 如連續的三個詞['used', 'against', 'early']
# 產生兩個樣本：against -> used, against -> early
batch, labels = generate_batch(batch_size=8, num_skips=2, skip_window=1)
for i in range(8):
  print(batch[i], reverse_dictionary[batch[i]],
        '->', labels[i, 0], reverse_dictionary[labels[i, 0]])



# 第四步: 建立模型.

batch_size = 128
embedding_size = 128  # 詞內嵌空間是128維的。即word2vec中的vec是一個128維的向量
skip_window = 1       # skip_window參數和之前保持一致
num_skips = 2         # num_skips參數和之前保持一致

# 在訓練過程中，會對模型進行驗證 
# 驗證的方法就是找出和某個詞最近的詞。
# 只對前valid_window的詞進行驗證，因為這些詞最常出現
valid_size = 16     # 每次驗證16個詞
valid_window = 100  # 這16個詞是在前100個最常見的詞中選出來的
valid_examples = np.random.choice(valid_window, valid_size, replace=False)

# 建構損失時選取的噪聲詞的數量
num_sampled = 64

graph = tf.Graph()

with graph.as_default():

  # 輸入的batch
  train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
  train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
  # 用於驗證的詞
  valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

  # 下面采用的某些函數還沒有gpu實現，所以我們只在cpu上定義模型
  with tf.device('/cpu:0'):
    # 定義1個embeddings變數，相當於一行儲存一個詞的embedding
    embeddings = tf.Variable(
        tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
    # 利用embedding_lookup可以輕松得到一個batch內的所有的詞內嵌
    embed = tf.nn.embedding_lookup(embeddings, train_inputs)

    # 建立兩個變數用於NCE Loss（即選取噪聲詞的二分類別損失）
    nce_weights = tf.Variable(
        tf.truncated_normal([vocabulary_size, embedding_size],
                            stddev=1.0 / math.sqrt(embedding_size)))
    nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

  # tf.nn.nce_loss會自動選取噪聲詞，並且形成損失。
  # 隨機選取num_sampled個噪聲詞
  loss = tf.reduce_mean(
      tf.nn.nce_loss(weights=nce_weights,
                     biases=nce_biases,
                     labels=train_labels,
                     inputs=embed,
                     num_sampled=num_sampled,
                     num_classes=vocabulary_size))

  # 得到loss後，我們就可以建構改善器了
  optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)

  # 計算詞和詞的相似度（用於驗證）
  norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
  normalized_embeddings = embeddings / norm
  # 找出和驗證詞的embedding並計算它們和所有單字的相似度
  valid_embeddings = tf.nn.embedding_lookup(
      normalized_embeddings, valid_dataset)
  similarity = tf.matmul(
      valid_embeddings, normalized_embeddings, transpose_b=True)

  # 變數起始化步驟
  init = tf.global_variables_initializer()



# 第五步：開始訓練
num_steps = 100001

with tf.Session(graph=graph) as session:
  # 起始化變數
  init.run()
  print('Initialized')

  average_loss = 0
  for step in xrange(num_steps):
    batch_inputs, batch_labels = generate_batch(
        batch_size, num_skips, skip_window)
    feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}

    # 改善一步
    _, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
    average_loss += loss_val

    if step % 2000 == 0:
      if step > 0:
        average_loss /= 2000
      # 2000個batch的平均損失
      print('Average loss at step ', step, ': ', average_loss)
      average_loss = 0

    # 每1萬步，我們進行一次驗證
    if step % 10000 == 0:
      # sim是驗證詞與所有詞之間的相似度
      sim = similarity.eval()
      # 一共有valid_size個驗證詞
      for i in xrange(valid_size):
        valid_word = reverse_dictionary[valid_examples[i]]
        top_k = 8  # 輸出最相鄰的8個詞語
        nearest = (-sim[i, :]).argsort()[1:top_k + 1]
        log_str = 'Nearest to %s:' % valid_word
        for k in xrange(top_k):
          close_word = reverse_dictionary[nearest[k]]
          log_str = '%s %s,' % (log_str, close_word)
        print(log_str)
  # final_embeddings是我們最後得到的embedding向量
  # 它的形狀是[vocabulary_size, embedding_size]
  # 每一行就代表著對應index詞的詞內嵌表示
  final_embeddings = normalized_embeddings.eval()



# Step 6: 可視化
# 可視化的圖片會儲存為“tsne.png”

def plot_with_labels(low_dim_embs, labels, filename='tsne.png'):
  assert low_dim_embs.shape[0] >= len(labels), 'More labels than embeddings'
  plt.figure(figsize=(18, 18))  # in inches
  for i, label in enumerate(labels):
    x, y = low_dim_embs[i, :]
    plt.scatter(x, y)
    plt.annotate(label,
                 xy=(x, y),
                 xytext=(5, 2),
                 textcoords='offset points',
                 ha='right',
                 va='bottom')

  plt.savefig(filename)

try:
  # pylint: disable=g-import-not-at-top
  from sklearn.manifold import TSNE
  import matplotlib
  matplotlib.use('agg')
  import matplotlib.pyplot as plt
  # 因為我們的embedding的大小為128維，沒有辦法直接可視化
  # 所以我們用t-SNE方法進行降維
  tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
  # 只畫出500個詞的位置
  plot_only = 500
  low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only, :])
  labels = [reverse_dictionary[i] for i in xrange(plot_only)]
  plot_with_labels(low_dim_embs, labels)

except ImportError:
  print('Please install sklearn, matplotlib, and scipy to show embeddings.')

