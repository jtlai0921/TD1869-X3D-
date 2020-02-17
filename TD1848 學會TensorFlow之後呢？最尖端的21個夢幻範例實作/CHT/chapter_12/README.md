### 12. RNN基本結構與Char RNN文字產生

**12.5.4 訓練模型與產生文字**

訓練產生英文的模型：
```
python train.py \
  --input_file data/shakespeare.txt \
  --name shakespeare \
  --num_steps 50 \
  --num_seqs 32 \
  --learning_rate 0.01 \
  --max_steps 20000
```

測試模型：
```
python sample.py \
  --converter_path model/shakespeare/converter.pkl \
  --checkpoint_path model/shakespeare/ \
  --max_length 1000
```

訓練寫詩模型：
```
python train.py \
  --use_embedding \
  --input_file data/poetry.txt \
  --name poetry \
  --learning_rate 0.005 \
  --num_steps 26 \
  --num_seqs 32 \
  --max_steps 10000
```


測試模型：
```
python sample.py \
  --use_embedding \
  --converter_path model/poetry/converter.pkl \
  --checkpoint_path model/poetry/ \
  --max_length 300
```

訓練產生C程式碼的模型：
```
python train.py \
  --input_file data/linux.txt \
  --num_steps 100 \
  --name linux \
  --learning_rate 0.01 \
  --num_seqs 32 \
  --max_steps 20000
```

測試模型：
```
python sample.py \
  --converter_path model/linux/converter.pkl \
  --checkpoint_path model/linux \
  --max_length 1000
```

#### 拓展閱讀

- 若果讀者想要深入了解RNN 的結構及其訓練方法，建議閱讀書籍 Deep Learning（Ian Goodfellow、Yoshua Bengio 和 Aaron Courville 所著）的第10章“Sequence Modeling: Recurrent and Recursive Nets”。 此外，http://karpathy.github.io/2015/05/21/rnn-effectiveness/ 中詳細地介紹了RNN 以及Char RNN 的原理，也是很好的閱讀材料。

- 若果讀者想要深入了解LSTM 的結構， 推薦閱讀 http://colah.github.io/posts/2015-08-Understanding-LSTMs/ 。有網友對這篇博文做了翻譯，位址為：http://blog.csdn.net/jerr__y/article/ details/58598296。

- 關於TensorFlow 中的RNN 實現，有興趣的讀者可以閱讀TensorFlow 原始程式進行詳細了解，位址為：https://github.com/tensorflow/tensorflow/ blob/master/ tensorflow/python/ops/rnn_cell_impl.py 。該原始程式檔案中有BasicRNNCell、BasicLSTMCell、RNNCell、LSTMCell 的實現。
