### 20. 深度強化研讀：Deep Q Learning

本節的程式來源於專案 https://github.com/carpedm20/deep-rl-tensorflow。

**20.2.1 安裝相依庫**

```
pip install gym[all] scipy tqdm
```

**20.2.2 訓練**

使用GPU訓練：
```
python main.py --network_header_type=nips --env_name=Breakout-v0 --use_gpu=True
```

使用CPU訓練：
```
python main.py --network_header_type=nips --env_name=Breakout-v0 --use_gpu=False
```

開啟TensorBoard：
```
tensorboard --logdir logs/
```

**20.2.3 測試**

測試在GPU上訓練的模型：

```
python main.py --network_header_type=nips --env_name=Breakout-v0 --use_gpu=True --is_train=False
```

測試在CPU上訓練的模型：
```
python main.py --network_header_type=nips --env_name=Breakout-v0 --use_gpu=True --is_train=True
```

在上述指令中加入--display=True選項，可以實時顯示游戲執行緒。

#### 拓展閱讀

- 本章主要介紹了深度強化研讀算法DQN，關於該算法的更多細節，可以參考論文Playing Atari with Deep Reinforcement Learning。

- 本章還介紹了OpenAI 的gym 庫，它可以為我們提供常用的強化學 習環境。讀者可以參考它的文件https://gym.openai.com/docs/ 了解 gym 庫的使用細節，此外還可以在https://gym.openai.com/envs/ 看到目前Gym 庫支援的所有環境。
