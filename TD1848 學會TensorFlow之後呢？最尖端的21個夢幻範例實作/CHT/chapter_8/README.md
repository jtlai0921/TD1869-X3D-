### 8. GAN與DCGAN入門

本節的程式來自於專案 https://github.com/carpedm20/DCGAN-tensorflow 。

**8.3.1 產生MNIST圖形**

下載MNIST資料集：
```
python download.py mnist
```

訓練：
```
python main.py --dataset mnist --input_height=28 --output_height=28 --train
```

產生圖形儲存在samples資料夾中。

**8.3.2 使用自己的資料集訓練**

在資料目錄chapter_8_data/中已經準備好了一個動漫人物圖示資料集faces.zip。在源程式碼的data目錄中再新增一個anime目錄（若果沒有data 目錄可以自行新增），並將faces.zip 中所有的圖形檔案解壓到anime 目錄中。

訓練指令：
```
python main.py --input_height 96 --input_width 96 \
  --output_height 48 --output_width 48 \
  --dataset anime --crop -–train \
  --epoch 300 --input_fname_pattern "*.jpg"
```

產生圖形儲存在samples資料夾中。


#### 拓展閱讀

- 本章只講了GAN 結構和訓練方法，在提出GAN 的原始論文 Generative Adversarial Networks 中，還有關於GAN 收斂性的理論證明以及更多實驗細節，讀者可以閱讀來深入瞭解GAN 的思想。

- 有關DCGAN的更多細節， 可以閱讀其論文Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks。

- 除了本章所講的GAN 和DCGAN 外，還有研究者對原始GAN 的損 失函數做了改進，改進後的模型可以在某些資料集上獲得更穩定的 產生效果，關聯的論文有：Wasserstein GAN、Least Squares Generative Adversarial Networks。

- 相比一般的神經網路，訓練GAN 往往會更加困難。Github 使用者 Soumith Chintala 收集了一份訓練GAN 的技巧清單：https://github.com/soumith/ganhacks ，在實作中很有幫助。
