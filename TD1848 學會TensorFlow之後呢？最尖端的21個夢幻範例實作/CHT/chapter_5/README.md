### 5. 深度研讀中的目的檢驗

**5.2.1 安裝TensorFlow Object Detection API**

參考5.2.1小節完成對應動作。

**5.2.3 訓練新的模型**

先在位址http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar 下載VOC2012資料集並解壓。

在專案的object_detection資料夾中新增voc目錄，並將解壓後的資料集覆制進來，最終形成的目錄為：

```
research/
  object_detection/
    voc/
      VOCdevkit/
        VOC2012/
          JPEGImages/
            2007_000027.jpg
            2007_000032.jpg
            2007_000033.jpg
            2007_000039.jpg
            2007_000042.jpg
            ………………
          Annotations/
            2007_000027.xml
            2007_000032.xml
            2007_000033.xml
            2007_000039.xml
            2007_000042.xml
            ………………
          ………………
```

在object_detection目錄中執行如下指令將資料集轉為tfrecord：

```
python create_pascal_tf_record.py --data_dir voc/VOCdevkit/ --year=VOC2012 --set=train --output_path=voc/pascal_train.record
python create_pascal_tf_record.py --data_dir voc/VOCdevkit/ --year=VOC2012 --set=val --output_path=voc/pascal_val.record
```

此外，將pascal_label_map.pbtxt 資料複製到voc 資料夾下：
```
cp data/pascal_label_map.pbtxt voc/
```

下載模型檔案http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_resnet_v2_atrous_coco_11_06_2017.tar.gz 並解壓，解壓後得到frozen_inference_graph.pb 、graph.pbtxt 、model.ckpt.data-00000-of-00001 、model.ckpt.index、model.ckpt.meta 5 個檔案。在voc資料夾中新增一個
pretrained 資料夾，並將這5個檔案複製進去。

複製一份config檔案：
```
cp samples/configs/faster_rcnn_inception_resnet_v2_atrous_pets.config \
  voc/voc.config
```

並在voc/voc.config中修改7處需要重新組態的地方（詳見書本）。

訓練模型的指令：
```
python train.py --train_dir voc/train_dir/ --pipeline_config_path voc/voc.config
```

使用TensorBoard：
```
tensorboard --logdir voc/train_dir/
```

**5.2.4 匯出模型並預測單張圖片**

執行(需要根據voc/train_dir/裡實際儲存的checkpoint，將1582改為合適的數值)：
```
python export_inference_graph.py \
  --input_type image_tensor \
  --pipeline_config_path voc/voc.config \
  --trained_checkpoint_prefix voc/train_dir/model.ckpt-1582
  --output_directory voc/export/
```

匯出的模型是voc/export/frozen_inference_graph.pb 檔案。

#### 拓展閱讀

- 本章提到的R-CNN、SPPNet、Fast R-CNN、Faster R-CNN 都是基於 區域的深度目的檢驗方法。可以按順序閱讀以下論文了解更多細節： Rich Feature Hierarchies for Accurate Object Detection and Semantic Segmentation (R-CNN) 、Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition（SPPNet）、Fast R-CNN （Fast R-CNN）、Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks（Faster R-CNN）。

- 限於篇幅，除了本章提到的這些方法外，還有一些有較高參考價值 的深度研讀目的檢驗方法，這裡同樣推薦一下關聯的論文：R-FCN: Object Detection via Region-based Fully Convolutional Networks （R-FCN）、You Only Look Once: Unified, Real-Time Object Detection （YOLO）、SSD: Single Shot MultiBox Detector（SSD）、YOLO9000: Better, Faster, Stronger（YOLO v2 和YOLO9000）等。
