### 4. Deep Dream

本節的程式碼參考了TensorFlow 原始程式中的範例程式[https://github.com/tensorflow/tensorflow/tree/master/tensorflow/examples/tutorials/deepdream](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/examples/tutorials/deepdream)，並做了適當修改。

**4.2.1 匯入Inception 模型**

在chapter_4_data/中或是網址https://storage.googleapis.com/download.tensorflow.org/models/inception5h.zip 下載解壓得到模型檔案tensorflow_inception_graph.pb，將該檔案覆制到目前資料夾中（即chapter_4/中）。

使用下面的指令載入模型並列印一些基礎訊息：
```
python load_inception.py
```

**4.2.2 產生原始的Deep Dream 圖形**

```
python gen_naive.py
```

**4.2.3 產生更大尺寸的Deep Dream 圖形**
```
python gen_multiscale.py
```

**4.2.4 產生更高質量的Deep Dream 圖形**
```
python gen_lapnorm.py
```

**4.2.5 最終的Deep Dream 模型**
```
python gen_deepdream.py
```
