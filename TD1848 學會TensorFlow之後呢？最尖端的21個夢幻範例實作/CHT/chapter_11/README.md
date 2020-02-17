### 11. CycleGAN與非配對圖形轉換

本節的程式來源於專案https://github.com/vanhuyz/CycleGAN-TensorFlow，並做了細微修改。

**11.2.1 下載資料集並訓練**

下載一個事先準備好的資料集：
```
bash download_dataset.sh apple2orange
```

將圖片轉換成tfrecords格式：

```
python build_data.py \
  --X_input_dir data/apple2orange/trainA \
  --Y_input_dir data/apple2orange/trainB \
  --X_output_file data/tfrecords/apple.tfrecords \
  --Y_output_file data/tfrecords/orange.tfrecords
```

訓練模型：
```
python train.py \
  --X data/tfrecords/apple.tfrecords \
  --Y data/tfrecords/orange.tfrecords \
  --image_size 256
```

開啟TensorBoard(需要將--logdir checkpoints/20170715-1622 中的目錄置換為自己機器中的對應目錄)：
```
tensorboard --logdir checkpoints/20170715-1622
```

匯出模型(同樣要注意將20170715-1622 置換為自己機器中的對應目錄)：
```
python export_graph.py \
  --checkpoint_dir checkpoints/20170715-1622 \
  --XtoY_model apple2orange.pb \
  --YtoX_model orange2apple.pb \
  --image_size 256
```

使用測試集中的圖片進行測試：
```
python inference.py \
  --model pretrained/apple2orange.pb \
  --input data/apple2orange/testA/n07740461_1661.jpg \
  --output data/apple2orange/output_sample.jpg \
  --image_size 256
```

轉換產生的圖片儲存在data/apple2orange/output_sample. jpg。

**11.2.2 使用自己的資料進行訓練**

在chapter_11_data/中，事先提供了一個資料集man2woman.zip。，解壓後共包括兩個資料夾：a_resized 和b_resized。將它們放到目錄~/datasets/man2woman/下。使用下面的指令將資料集轉為tfrecords：
```
python build_data.py \
  --X_input_dir ~/datasets/man2woman/a_resized/ \
  --Y_input_dir ~/datasets/man2woman/b_resized/ \
  --X_output_file ~/datasets/man2woman/man.tfrecords \
  --Y_output_file ~/datasets/man2woman/woman.tfrecords
```

訓練：
```
python train.py \
  --X ~/datasets/man2woman/man.tfrecords \
  --Y ~/datasets/man2woman/woman.tfrecords \
  --image_size 256
```

匯出模型和測試圖片的指令可參考11.2.1。

#### 拓展閱讀

- 本章主要講了模型CycleGAN ， 讀者可以參考論文Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks 了解更多細節

- CycleGAN 不需要成對資料就可以訓練，具有較強的通用性，由此產生了大量有創意的套用，例如男女互換（即本章所介紹的）、貓狗互換、利用手繪地圖復原古代城市等。可以參考https://zhuanlan.zhihu.com/p/28342644 以及https://junyanz.github.io/CycleGAN/ 了解這些有趣的實驗

- CycleGAN 可以將將某一類別圖片轉換成另外一類別圖片。若果想要把一張圖片轉為另外K類別圖片，就需要訓練K個CycleGAN，這是比較麻煩的。對此，一種名為StarGAN 的方法改進了CycleGAN， 可以只用一個模型完成K類別圖片的轉換，有興趣的讀者可以參閱其論文StarGAN: Unified Generative Adversarial Networks for Multi-Domain Image-to-Image Translation。

- 若果讀者還想研讀更多和GAN 關聯的模型， 可以參考 https://github.com/hindupuravinash/the-gan-zoo 。這裡列出了迄今幾乎所有的名字中帶有“GAN”的模型和對應的論文。
