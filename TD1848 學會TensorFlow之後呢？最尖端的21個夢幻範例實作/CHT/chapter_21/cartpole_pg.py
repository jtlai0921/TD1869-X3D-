# coding:utf-8
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import gym

# gym環境
env = gym.make('CartPole-v0')

# 超參數
D = 4  # 輸入層神經元個數
H = 10  # 隱層神經元個數
batch_size = 5  # 一個batch中有5個episode，即5次游戲
learning_rate = 1e-2  # 研讀率
gamma = 0.99  # 獎勵折扣率gamma


# 定義policy網路
# 輸入觀察值，輸出右移的機率
observations = tf.placeholder(tf.float32, [None, D], name="input_x")
W1 = tf.get_variable("W1", shape=[D, H],
                     initializer=tf.contrib.layers.xavier_initializer())
layer1 = tf.nn.relu(tf.matmul(observations, W1))
W2 = tf.get_variable("W2", shape=[H, 1],
                     initializer=tf.contrib.layers.xavier_initializer())
score = tf.matmul(layer1, W2)
probability = tf.nn.sigmoid(score)

# 定義和訓練、loss有關的變數
tvars = tf.trainable_variables()
input_y = tf.placeholder(tf.float32, [None, 1], name="input_y")
advantages = tf.placeholder(tf.float32, name="reward_signal")

# 定義loss函數
loglik = tf.log(input_y * (input_y - probability) + (1 - input_y) * (input_y + probability))
loss = -tf.reduce_mean(loglik * advantages)
newGrads = tf.gradients(loss, tvars)

# 改善器、梯度。
adam = tf.train.AdamOptimizer(learning_rate=learning_rate)
W1Grad = tf.placeholder(tf.float32, name="batch_grad1")
W2Grad = tf.placeholder(tf.float32, name="batch_grad2")
batchGrad = [W1Grad, W2Grad]
updateGrads = adam.apply_gradients(zip(batchGrad, tvars))


def discount_rewards(r):
    """
    輸入：
        1維的float型態陣列，表示每個時刻的獎勵
    輸出：
        計算折扣率gamma後的期望獎勵
    """
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size)):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r


xs, hs, dlogps, drs, ys, tfps = [], [], [], [], [], []
running_reward = None
reward_sum = 0
episode_number = 1
total_episodes = 10000
init = tf.global_variables_initializer()

# 開始訓練
with tf.Session() as sess:
    rendering = False
    sess.run(init)
    # observation是環境的起始觀察量（輸入神經網路的值）
    observation = env.reset()

    # gradBuffer會儲存梯度，此處做一起始化
    gradBuffer = sess.run(tvars)
    for ix, grad in enumerate(gradBuffer):
        gradBuffer[ix] = grad * 0

    while episode_number <= total_episodes:

        # 當一個batch內的平均獎勵達到180以上時，顯示游戲視窗
        if reward_sum / batch_size > 180 or rendering is True:
            env.render()
            rendering = True

        # 輸入神經網路的值
        x = np.reshape(observation, [1, D])

        # action=1表示向右移
        # action=0表示向左移
        # tfprob為網路輸出的向右走的機率
        tfprob = sess.run(probability, feed_dict={observations: x})
        # np.random.uniform()為0~1之間的隨機數
        # 當它小於tfprob時，就采取右移策略，反之左移
        action = 1 if np.random.uniform() < tfprob else 0

        # xs記錄每一步的觀察量，ys記錄每一步采取的策略
        xs.append(x)
        y = 1 if action == 0 else 0
        ys.append(y)

        # 執行action
        observation, reward, done, info = env.step(action)
        reward_sum += reward

        # drs記錄每一步的reward
        drs.append(reward)

        # 一局游戲結束
        if done:
            episode_number += 1
            # 將xs、ys、drs從list變成numpy陣列形式
            epx = np.vstack(xs)
            epy = np.vstack(ys)
            epr = np.vstack(drs)
            tfp = tfps
            xs, hs, dlogps, drs, ys, tfps = [], [], [], [], [], []  # reset array memory

            # 對epr計算期望獎勵
            discounted_epr = discount_rewards(epr)
            # 對期望獎勵做歸一化
            discounted_epr -= np.mean(discounted_epr)
            discounted_epr //= np.std(discounted_epr)

            # 將梯度存到gradBuffer中
            tGrad = sess.run(newGrads, feed_dict={observations: epx, input_y: epy, advantages: discounted_epr})
            for ix, grad in enumerate(tGrad):
                gradBuffer[ix] += grad

            # 每batch_size局游戲，就將gradBuffer中的梯度真正更新到policy網路中
            if episode_number % batch_size == 0:
                sess.run(updateGrads, feed_dict={W1Grad: gradBuffer[0], W2Grad: gradBuffer[1]})
                for ix, grad in enumerate(gradBuffer):
                    gradBuffer[ix] = grad * 0

                # 列印一些訊息
                print('Episode: %d ~ %d Average reward: %f.  ' % (episode_number - batch_size + 1, episode_number, reward_sum // batch_size))

                # 當我們在batch_size游戲中平均能拿到200的獎勵，就停止訓練
                if reward_sum // batch_size >= 200:
                    print("Task solved in", episode_number, 'episodes!')
                    break

                reward_sum = 0

            observation = env.reset()

print(episode_number, 'Episodes completed.')
