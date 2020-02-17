### 9. pix2pix模型與自動著色技術

本節的程式來源於專案 https://github.com/affinelayer/pix2pix-tensorflow 。

**9.3.1 執行已有的資料集**

下載Facades資料集：
```
python tools/download-dataset.py facades
```

訓練：
```
python pix2pix.py \
  --mode train \
  --output_dir facades_train \
  --max_epochs 200 \
  --input_dir facades/train \
  --which_direction BtoA
```

測試：
```
python pix2pix.py \
  --mode test \
  --output_dir facades_test \
  --input_dir facades/val \
  --checkpoint facades_train
```

結果在facades_test資料夾中。


**9.4.1 為食物圖片著色**


在chapter_9_data/中提供的food_resized.zip 檔案解壓到目錄~/datasets/colorlization/下，最終形成的檔案
夾結構應該是：

```
~/datasets
  colorlization/
    food_resized/
      train/
      val/
```

訓練指令：
```
python pix2pix.py \
--mode train \
--output_dir colorlization_food \
--max_epochs 70 \
--input_dir ~/datasets/colorlization/food_resized/train \
--lab_colorization
```

測試指令：
```
python pix2pix.py \
  --mode test \
  --output_dir colorlization_food_test \
  --input_dir ~/datasets/colorlization/food_resized/val \
  --checkpoint colorlization_food
```

結果在colorlization_food_test資料夾中。

**9.4.2 為動漫圖片著色**

將chapter_9_data/中提供的動漫圖形資料集anime_reized.zip 解壓到~/datasets/colorlization/目錄下，形成的資料夾結構為：

```
~/datasets
  colorlization/
    anime_resized/
      train/
      val/
```

訓練指令：
```
python pix2pix.py \
  --mode train \
  --output_dir colorlization_anime \
  --max_epochs 5 \
  --input_dir ~/datasets/colorlization/anime_resized/train \
  --lab_colorization
```

測試指令：
```
python pix2pix.py \
  --mode test \
  --output_dir colorlization_anime_test \
  --input_dir ~/datasets/colorlization/anime_resized/val \
  --checkpoint colorlization_anime
```

結果在colorlization_anime_test資料夾中。


#### 拓展閱讀

- 本章主要講了cGAN 和pix2pix 兩個模型。讀者可以參考它們的原始 論文Conditional Generative Adversarial Nets 和Image-to-Image Translation with Conditional Adversarial Networks 研讀更多細節。

- 針對pix2pix 模型，這裡有一個線上示範Demo，已經預訓練好了多 種模型， 可以在瀏覽器中直接經驗pix2pix 模型的效果： https://affinelayer.com/pixsrv/ 。
