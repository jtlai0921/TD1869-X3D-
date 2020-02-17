### 1. MNIST機器研讀入門

**1.1.1 簡介**

下載MNIST資料集，並列印一些基本訊息：
```
python download.py
```

**1.1.2 實驗：將MNIST資料集儲存為圖片**

```
python save_pic.py
```

**1.1.3 圖形標簽的獨熱表示**

列印MNIST資料集中圖片的標簽：
```
python label.py
```

**1.2.1 Softmax 回歸**

```
python softmax_regression.py
```

**1.2.2 兩層卷冊積網路分類別**
```
python convolutional.py
```

#### 可能出現的錯誤

下載資料集時可能出現網路問題，可以用下面兩種方法中的一種解決：1. 使用合適的代理 2.在MNIST的官方網站上下載檔案train-images-idx3-ubyte.gz、train-labels-idx1-ubyte.gz、t10k-images-idx3-ubyte.gz、t10k-labels-idx1-ubyte.gz，並將它們儲存在MNIST_data/資料夾中。


#### 拓展閱讀

- 本章介紹的MNIST 資料集經常被用來檢驗機器研讀模型的效能，在它的官網（位址：http://yann.lecun.com/exdb/mnist/ ）中，可以找到多達68 種模型在該資料集上的準確率資料，內含對應的論文出處。這些模型內含線性分類別器、K 近鄰方法、普通的神經網路、卷冊積神經網路等。
- 本章的兩個MNIST 程式實際上來自於TensorFlow 官方的兩個新手教學，位址為https://www.tensorflow.org/get_started/mnist/beginners 和 https://www.tensorflow.org/get_started/mnist/pros 。讀者可以將本書的內容和官方的教學對照起來進行閱讀。這兩個新手教學的中文版位址為http://www.tensorfly.cn/tfdoc/tutorials/mnist_beginners.html 和http://www.tensorfly.cn/tfdoc/tutorials/mnist_pros.html。
- 本章簡要介紹了TensorFlow 的tf.Tensor 類別。tf.Tensor 類別是TensorFlow的核心類別，常用的占位符（tf.placeholder）、變數（tf.Variable）都可以看作特殊的Tensor。讀者可以參閱https://www.tensorflow.org/programmers_guide/tensors 來更深入地研讀它的原理。
- 常用tf.Variable 類別來儲存模型的參數， 讀者可以參閱[https://www.tensorflow.org/programmers_guide/variables](https://www.tensorflow.org/programmers_guide/variables) 詳細了解它的執行機制， 文件的中文版位址為http://www.tensorfly.cn/tfdoc/how_tos/ variables.html。
- 只有透過階段（Session）才能計算出tf.Tensor 的值。強烈建議讀者 在研讀完tf.Tensor 和tf.Variable 後，閱讀https://www.tensorflow.org/programmers_guide/graphs 中的內容，該文件描述了TensorFlow 中 計算圖和階段的基本執行原理，對瞭解TensorFlow 的底層原理有很 大幫助。
