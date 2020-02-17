### 3. 打造自己的圖形識別模型

#### 執行方法

**3.2 資料準備**

首先需要將資料轉換成tfrecord的形式。在data_prepare資料夾下，執行：
```
python data_convert.py -t pic/ \
  --train-shards 2 \
  --validation-shards 2 \
  --num-threads 2 \
  --dataset-name satellite
```
這樣在pic資料夾下就會產生4個tfrecord檔案和1個label.txt檔案。

**3.3.2 定義新的datasets 檔案**

參考3.3.2小節對Slim原始程式做修改。

**3.3.3 準備訓練資料夾**

在slim資料夾下新增一個satellite目錄。在這個目錄下做下面幾件事情：
- 新增一個data 目錄，並將第3.2中準備好的5個轉換好格式的訓練資料複製進去。
- 新增一個空的train_dir目錄，用來儲存訓練過程中的日志和模型。
- 新增一個pretrained目錄，在slim的GitHub頁面找到Inception V3 模型的下載位址http://download.tensorflow.org/models/inception_v3_2016_08_28.tar.gz 下載並解壓後，會得到一個inception_v3.ckpt 檔案，將該檔案複製到pretrained 目錄下（這個檔案在chapter_3_data/檔案中也提供了）

**3.3.4 開始訓練**

（在slim資料夾下執行）訓練Logits層：
```
python train_image_classifier.py \
  --train_dir=satellite/train_dir \
  --dataset_name=satellite \
  --dataset_split_name=train \
  --dataset_dir=satellite/data \
  --model_name=inception_v3 \
  --checkpoint_path=satellite/pretrained/inception_v3.ckpt \
  --checkpoint_exclude_scopes=InceptionV3/Logits,InceptionV3/AuxLogits \
  --trainable_scopes=InceptionV3/Logits,InceptionV3/AuxLogits \
  --max_number_of_steps=100000 \
  --batch_size=32 \
  --learning_rate=0.001 \
  --learning_rate_decay_type=fixed \
  --save_interval_secs=300 \
  --save_summaries_secs=2 \
  --log_every_n_steps=10 \
  --optimizer=rmsprop \
  --weight_decay=0.00004
```

訓練所有層：
```
python train_image_classifier.py \
  --train_dir=satellite/train_dir \
  --dataset_name=satellite \
  --dataset_split_name=train \
  --dataset_dir=satellite/data \
  --model_name=inception_v3 \
  --checkpoint_path=satellite/pretrained/inception_v3.ckpt \
  --checkpoint_exclude_scopes=InceptionV3/Logits,InceptionV3/AuxLogits \
  --max_number_of_steps=100000 \
  --batch_size=32 \
  --learning_rate=0.001 \
  --learning_rate_decay_type=fixed \
  --save_interval_secs=300 \
  --save_summaries_secs=10 \
  --log_every_n_steps=1 \
  --optimizer=rmsprop \
  --weight_decay=0.00004
```

**3.3.6 驗證模型準確率**

在slim資料夾下執行：
```
python eval_image_classifier.py \
  --checkpoint_path=satellite/train_dir \
  --eval_dir=satellite/eval_dir \
  --dataset_name=satellite \
  --dataset_split_name=validation \
  --dataset_dir=satellite/data \
  --model_name=inception_v3
```

**3.3.7 TensorBoard 可視化與超參數選取**

開啟TensorBoard：
```
tensorboard --logdir satellite/train_dir
```

**3.3.8 匯出模型並對單張圖片進行識別**

在slim資料夾下執行：
```
python export_inference_graph.py \
  --alsologtostderr \
  --model_name=inception_v3 \
  --output_file=satellite/inception_v3_inf_graph.pb \
  --dataset_name satellite
```

在chapter_3資料夾下執行（需將5271改成train_dir中儲存的實際的模型訓練步數）：
```
python freeze_graph.py \
  --input_graph slim/satellite/inception_v3_inf_graph.pb \
  --input_checkpoint slim/satellite/train_dir/model.ckpt-5271 \
  --input_binary true \
  --output_node_names InceptionV3/Predictions/Reshape_1 \
  --output_graph slim/satellite/frozen_graph.pb
```

執行匯出模型分類別單張圖片：
```
python classify_image_inception_v3.py \
  --model_path slim/satellite/frozen_graph.pb \
  --label_path data_prepare/pic/label.txt \
  --image_file test_image.jpg
```


#### 拓展閱讀

- TensorFlow Slim 是TensorFlow 中用於定義、訓練和驗證復雜網路的 高層API。官方已經使用TF-Slim 定義了一些常用的圖形識別模型， 如AlexNet、VGGNet、Inception模型、ResNet等。本章介紹的Inception V3 模型也是其中之一， 詳細文件請參考： https://github.com/tensorflow/models/tree/master/research/slim。
- 在第3.2節中，將圖片資料轉換成了TFRecord檔案。TFRecord 是 TensorFlow 提供的用於高速讀取資料的檔案格式。讀者可以參考博文（ http://warmspringwinds.github.io/tensorflow/tf-slim/2016/12/21/tfrecords-guide/ ）詳細了解如何將資料轉為TFRecord 檔案，以及 如何從TFRecord 檔案中讀取資料。
- Inception V3 是Inception 模型（即GoogLeNet）的改進版，可以參考論文Rethinking the Inception Architecture for Computer Vision 了解 其結構細節。
