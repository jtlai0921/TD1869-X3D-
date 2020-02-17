### 6. 人臉檢驗和人臉識別

本節的程式來自於專案https://github.com/davidsandberg/facenet 。

**6.4.1 專案環境設定**

參考6.4.1小節。

**6.4.2 LFW 人臉資料庫**

在位址http://vis-www.cs.umass.edu/lfw/lfw.tgz 下載lfw資料集，並解壓到~/datasets/中：
```
cd ~/datasets
mkdir -p lfw/raw
tar xvf ~/Downloads/lfw.tgz -C ./lfw/raw --strip-components=1
```

**6.4.3 LFW 資料庫上的人臉檢驗和對齊**

對LFW進行人臉檢驗和對齊：

```
python src/align/align_dataset_mtcnn.py \
  ~/datasets/lfw/raw \
  ~/datasets/lfw/lfw_mtcnnpy_160 \
  --image_size 160 --margin 32 \
  --random_order
```

在輸出目錄~/datasets/lfw/lfw_mtcnnpy_160中可以找到檢驗、對齊後裁剪好的人臉。

**6.4.4 使用已有模型驗證LFW 資料庫準確率**

在百度網路硬碟的chapter_6_data/目錄或是位址https://drive.google.com/file/d/0B5MzpY9kBtDVZ2RpVDYwWmxoSUk 下載解壓得到4個模型資料夾，將它們覆制到~/models/facenet/20170512-110547/中。

之後執行程式碼：
```
python src/validate_on_lfw.py \
  ~/datasets/lfw/lfw_mtcnnpy_160 \
  ~/models/facenet/20170512-110547/
```

即可驗證該模型在已經裁剪好的lfw資料集上的準確率。

**6.4.5 在自己的資料上使用已有模型**

計算人臉兩兩之間的距離：
```
python src/compare.py \
  ~/models/facenet/20170512-110547/ \
  ./test_imgs/1.jpg ./test_imgs/2.jpg ./test_imgs/3.jpg
```

**6.4.6 重新訓練新模型**

以CASIA-WebFace資料集為例，讀者需自行申請該資料集，申請位址為http://www.cbsr.ia.ac.cn/english/CASIA-WebFace-Database.html 。獲得CASIA-WebFace 資料集後，將它解壓到~/datasets/casia/raw 目錄中。此時資料夾~/datasets/casia/raw/中的資料結構應該類別似於：
```
0000045
  001.jpg
  002.jpg
  003.jpg
  ……
0000099
  001.jpg
  002.jpg
  003.jpg
  ……
……
```

先用MTCNN進行檢驗和對齊：
```
python src/align/align_dataset_mtcnn.py \
  ~/datasets/casia/raw/ \
  ~/datasets/casia/casia_maxpy_mtcnnpy_182 \
  --image_size 182 --margin 44
```

再進行訓練：
```
python src/train_softmax.py \
  --logs_base_dir ~/logs/facenet/ \
  --models_base_dir ~/models/facenet/ \
  --data_dir ~/datasets/casia/casia_maxpy_mtcnnpy_182 \
  --image_size 160 \
  --model_def models.inception_resnet_v1 \
  --lfw_dir ~/datasets/lfw/lfw_mtcnnpy_160 \
  --optimizer RMSPROP \
  --learning_rate -1 \
  --max_nrof_epochs 80 \
  --keep_probability 0.8 \
  --random_crop --random_flip \
  --learning_rate_schedule_file
  data/learning_rate_schedule_classifier_casia.txt \
  --weight_decay 5e-5 \
  --center_loss_factor 1e-2 \
  --center_loss_alfa 0.9
```

開啟TensorBoard的指令(<開始訓練時間>需要進行置換)：
```
tensorboard --logdir ~/logs/facenet/<開始訓練時間>/
```

#### 拓展閱讀

- MTCNN是常用的人臉檢驗和人臉對齊模型，讀者可以參考論文Joint Face Detection and Alignment using Multi-task Cascaded Convolutional Networks 了解其細節。

- 訓練人臉識別模型通常需要包括大量人臉圖片的訓練資料集，常用 的人臉資料集有CAISA-WebFace（http://www.cbsr.ia.ac.cn/english/CASIA-WebFace-Database.html ）、VGG-Face（http://www.robots.ox.ac.uk/~vgg/data/vgg_face/ ）、MS-Celeb-1M（https://www.microsoft.com/en-us/research/project/ms-celeb-1m-challenge-recognizing-one-millioncelebrities-real-world/ ）、MegaFace（ http://megaface.cs.washington.edu/ ）。更多資料集可以參考網站：http://www.face-rec.org/databases

- 關於Triplet Loss 的詳細介紹，可以參考論文FaceNet: A Unified Embedding for Face Recognition and Clustering，關於Center Loss 的 詳細介紹，可以參考論文A Discriminative Feature Learning Approach for Deep Face Recognition。
