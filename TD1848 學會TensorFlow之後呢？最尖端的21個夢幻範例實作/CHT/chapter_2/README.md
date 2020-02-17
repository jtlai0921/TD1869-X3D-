### 2. CIFAR-10與ImageNet圖形識別

**2.1.2 下載CIFAR-10 資料**

```
python cifar10_download.py
```

**2.1.3 TensorFlow 的資料讀取機制**

實驗指令稿：
```
python test.py
```

**2.1.4 實驗：將CIFAR-10 資料集儲存為圖片形式**

```
python cifar10_extract.py
```

**2.2.3 訓練模型**

```
python cifar10_train.py --train_dir cifar10_train/ --data_dir cifar10_data/
```

**2.2.4 在TensorFlow 中檢視訓練進度**
```
tensorboard --logdir cifar10_train/
```

**2.2.5 測試模型效果**
```
python cifar10_eval.py --data_dir cifar10_data/ --eval_dir cifar10_eval/ --checkpoint_dir cifar10_train/
```

使用TensorBoard檢視效能驗證情況：
```
tensorboard --logdir cifar10_eval/ --port 6007
```


#### 拓展閱讀

- 關於CIFAR-10 資料集， 讀者可以存取它的官方網站https://www.cs.toronto.edu/~kriz/cifar.html 了解更多細節。此外， 網站 http://rodrigob.github.io/are_we_there_yet/build/classification_datasets_results.html#43494641522d3130 中收集了在CIFAR-10 資料集上表 現最好的許多模型，內含這些模型對應的論文。
- ImageNet 資料集上的表現較好的幾個著名的模型是深度研讀的基礎， 值得仔細研讀。建議先閱讀下面幾篇論文：ImageNet Classification with Deep Convolutional Neural Networks（AlexNet 的提出）、Very Deep Convolutional Networks for Large-Scale Image Recognition （VGGNet）、Going Deeper with Convolutions（GoogLeNet）、Deep Residual Learning for Image Recognition（ResNet）
- 在第2.1.3 節中，簡要介紹了TensorFlow的一種資料讀入機制。事實上，目前在TensorFlow 中讀入資料大致有三種方法：（1）用占位符（即placeholder）讀入，這種方法比較簡單；（2）用佇列的形式建立檔案到Tensor的映射；（3）用Dataset API 讀入資料，Dataset API 是TensorFlow 1.3 版本新引入的一種讀取資料的機制，可以參考這 篇中文教學：https://zhuanlan.zhihu.com/p/30751039。
