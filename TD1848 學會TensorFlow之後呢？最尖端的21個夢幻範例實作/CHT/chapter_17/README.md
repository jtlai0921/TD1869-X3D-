### 17. 秀圖說話：將圖形轉為文字

**17.2.2 環境準備**

機器中沒有Bazel的需要安裝Bazel，這裡以Ubuntu系統為例，其他系統可以參考其官方網站https://docs.bazel.build/versions/master/install.html 進行安裝。

在Ubuntu 系統上安裝Bazel，首先要加入Bazel 對應的源：

```
echo "deb [arch=amd64] http://storage.googleapis.com/bazel-apt stable jdk1.8" | sudo tee /etc/apt/sources.list.d/bazel.list
curl https://bazel.build/bazel-release.pub.gpg | sudo apt-key add -
```

apt-get安裝：
```
sudo apt-get update && sudo apt-get install bazel
```

此外還需要安裝nltk：
```
pip install nltk
```

**17.2.3 編譯和資料準備**

編譯原始程式：
```
bazel build //im2txt:download_and_preprocess_mscoco
bazel build -c opt //im2txt/...
bazel build -c opt //im2txt:run_inference
```

下載訓練資料(請確保網路暢通，並確保至少有150GB 的硬碟空間可
以使用)：
```
bazel-bin/im2txt/download_and_preprocess_mscoco "data/mscoco"
```

下載http://download.tensorflow.org/models/inception_v3_2016_08_28.tar.gz ，解壓後得到inception_v3.ckpt。在data目錄下新增一個pretrained目錄，並將inception_v3.ckpt複製進去。

最後，在data 目錄下新增model 資料夾。並在該目錄下新增train 和eval 兩個資料夾，這兩個資料夾分別用來儲存訓練時的模型、日志和驗證時的日志。最終，資料夾結構應該是：
```
im2txt/
  data/
    mscoco/
    pretrained/
      inception_v3.ckpt
    model/
      train/
      eval/
```

**17.2.4 訓練和驗證**

訓練：
```
bazel-bin/im2txt/train \
  --input_file_pattern="data/mscoco/train-?????-of-00256" \
  --inception_checkpoint_file="data/pretrained/inception_v3.ckpt" \
  --train_dir="data/model/train" \
  --train_inception=false \
  --number_of_steps=1000000
```

開啟TensorBoard：
```
tensorboard –logdir data/model/train
```

驗證困惑度指標：
```
bazel-bin/im2txt/evaluate \
  --input_file_pattern="data/mscoco/val-?????-of-00004" \
  --checkpoint_dir="data/model/train" \
  --eval_dir="data/model/eval"
```

開啟TensorBoard 觀察驗證資料集上困惑度的變化：
```
tensorboard --logdir data/model/eval
```


**17.2.5 測試單張圖片**

```
bazel-bin/im2txt/run_inference \
  --checkpoint_path=data/model/train \
  --vocab_file=data/mscoco/word_counts.txt \
  --input_files=data/test.jpg
```

#### 拓展閱讀

- Image Caption 是一項仍在不斷發展的新技術，除了本章提到的論文 Show and Tell: A Neural Image Caption Generator、Neural machine translation by jointly learning to align and translate、What Value Do Explicit High Level Concepts Have in Vision to Language Problems? 外，還可閱讀Mind’s Eye: A Recurrent Visual Representation for Image Caption Generation、From Captions to Visual Concepts and Back 等論 文，了解其更多發展細節。
