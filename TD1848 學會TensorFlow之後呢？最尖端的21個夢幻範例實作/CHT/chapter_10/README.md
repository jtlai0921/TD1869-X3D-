### 10. 超解析度：如何讓圖形變得更清晰

本節的程式來源於專案 https://github.com/affinelayer/pix2pix-tensorflow 。

**10.1.1 去除錯誤圖片**

在位址http://msvocds.blob.core.windows.net/coco2014/train2014.zip 下載COCO資料集，將所有圖片檔案放在目錄~/datasets/super-resolution/mscoco下。使用chapter_10中的delete_broken_img.py指令稿移除一些錯誤圖形：

```
python delete_broken_img.py -p ~/datasets/super-resolution/mscoco/
```

**10.1.2 將圖形裁剪到統一大小**

接著將圖形縮放到統一大小：
```
python tools/process.py \
  --input_dir ~/datasets/super-resolution/mscoco/ \
  --operation resize \
  --output_dir ~/datasets/super-resolution/mscoco/resized
```

**10.1.3 為程式碼加入新的動作**

遵循 10.1.3 為程式碼加入新的blur動作，然後對圖形進行模糊處理：
```
python tools/process.py --operation blur \
  --input_dir ~/datasets/super-resolution/mscoco_resized/ \
  --output_dir ~/datasets/super-resolution/mscoco_blur/
```

合並圖形：
```
python tools/process.py \
  --input_dir ~/datasets/super-resolution/mscoco_resized/ \
  --b_dir ~/datasets/super-resolution/mscoco_blur/ \
  --operation combine \
  --output_dir ~/datasets/super-resolution/mscoco_combined/
```

劃分訓練集和測試集：
```
python tools/split.py \
  --dir ~/datasets/super-resolution/mscoco_combined/
```

模型訓練：
```
python pix2pix.py --mode train \
  --output_dir super_resolution \
  --max_epochs 20 \
  --input_dir ~/datasets/super-resolution/mscoco_combined/train \
  --which_direction BtoA
```

模型測試：
```
python pix2pix.py --mode test \
--output_dir super_resolution_test \
--input_dir ~/datasets/super-resolution/mscoco_combined/val \
--checkpoint super_resolution/
```

結果在super_resolution_test資料夾中。
