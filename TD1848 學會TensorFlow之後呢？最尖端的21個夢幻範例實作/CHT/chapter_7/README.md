### 7. 圖形風格遷移

**7.2.1 使用預訓練模型**

在chapter_7_data/ 中提供了7 個預訓練模型： wave.ckpt-done 、cubist.ckpt-done、denoised_starry.ckpt-done、mosaic.ckpt-done、scream.ckpt-done、feathers.ckpt-done。

以wave.ckpt-done的為例，在chapter_7/中新增一個models 檔案
夾， 然後把wave.ckpt-done複製到這個資料夾下，執行指令：
```
python eval.py --model_file models/wave.ckpt-done --image_file img/test.jpg
```

成功風格化的圖形會被寫到generated/res.jpg。

**7.2.2 訓練自己的模型**

準備工作：

- 在位址http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz 下載VGG16模型，將下載到的壓縮檔解壓後會得到一個vgg16.ckpt 檔案。在chapter_7/中新增一個資料夾pretrained，並將vgg16.ckpt 複製到pretrained 資料夾中。最後的檔案路徑是pretrained/vgg16.ckpt。這個vgg16.ckpt檔案在chapter_7_data/中也有提供。

- 在位址http://msvocds.blob.core.windows.net/coco2014/train2014.zip 下載COCO資料集。將該資料集解壓後會得到一個train2014 資料夾，其中應該含有大量jpg 格式的圖片。在chapter_7中建立到這個資料夾的符號連結：
```
ln –s <到train2014 資料夾的路徑> train2014
```

訓練wave模型：
```
python train.py -c conf/wave.yml
```

開啟TensorBoard：
```
tensorboard --logdir models/wave/
```

訓練中儲存的模型在資料夾models/wave/中。

#### 拓展閱讀

- 關於第7.1.1 節中介紹的原始的圖形風格遷移算法，可以參考論文A Neural Algorithm of Artistic Style 進一步了解其細節。關於第7.1.2 節 中介紹的快速風格遷移， 可以參考論文Perceptual Losses for Real-Time Style Transfer and Super-Resolution。

- 在訓練模型的過程中，用Instance Normalization 代替了常用的Batch Normalization，這可以提昇模型產生的圖片質量。關於Instance Normalization 的細節，可以參考論文Instance Normalization: The Missing Ingredient for Fast Stylization。

- 盡管快速遷移可以在GPU 下實時產生風格化圖片，但是它還有一個 很大的局限性，即需要事先為每一種風格訓練單獨的模型。論文 Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization 中提出了一種“Arbitrary Style Transfer”算法，可以 為任意風格實時產生風格化圖片，讀者可以參考該論文了解其實現 細節。
