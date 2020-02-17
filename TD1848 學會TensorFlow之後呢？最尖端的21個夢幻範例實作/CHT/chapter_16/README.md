### 16. 神經網路機器翻譯技術

**16.3.1 範例：將越南語翻譯為英語**

下載平行語料庫：
```
nmt/scripts/download_iwslt15.sh /tmp/nmt_data
```

訓練：
```
mkdir /tmp/nmt_model
python -m nmt.nmt \
  --src=vi --tgt=en \
  --vocab_prefix=/tmp/nmt_data/vocab \
  --train_prefix=/tmp/nmt_data/train \
  --dev_prefix=/tmp/nmt_data/tst2012 \
  --test_prefix=/tmp/nmt_data/tst2013 \
  --out_dir=/tmp/nmt_model \
  --num_train_steps=12000 \
  --steps_per_stats=100 \
  --num_layers=2 \
  --num_units=128 \
  --dropout=0.2 \
  --metrics=bleu
```

測試時，建立一個/tmp/my_infer_file.vi 檔案， 並將/tmp/nmt_data/tst2013.vi 中的越南敘述子複製一些到/tmp/my_infer_file.vi 裡，接著使用下面的指令產生其英語翻譯：
```
python -m nmt.nmt \
--out_dir=/tmp/nmt_model \
--inference_input_file=/tmp/my_infer_file.vi \
--inference_output_file=/tmp/nmt_model/output_infer
```

翻譯之後的結果在/tmp/nmt_model/output_infer。

訓練一個帶有注意力機制的模型：
```
mkdir /tmp/nmt_attention_model
python -m nmt.nmt \
  --attention=scaled_luong \
  --src=vi --tgt=en \
  --vocab_prefix=/tmp/nmt_data/vocab \
  --train_prefix=/tmp/nmt_data/train \
  --dev_prefix=/tmp/nmt_data/tst2012 \
  --test_prefix=/tmp/nmt_data/tst2013 \
  --out_dir=/tmp/nmt_attention_model \
  --num_train_steps=12000 \
  --steps_per_stats=100 \
  --num_layers=2 \
  --num_units=128 \
  --dropout=0.2 \
  --metrics=bleu
```

測試模型：
```
python -m nmt.nmt \
--out_dir=/tmp/nmt_attention_model \
--inference_input_file=/tmp/my_infer_file.vi \
--inference_output_file=/tmp/nmt_attention_model/output_infer
```

產生的翻譯會被儲存在/tmp/nmt_attention_model/output_infer 檔案中。


**16.3.2 建構中英翻譯引擎**

在chapter_16_data 中提供了一份整理好的中英平行語料資料，共分為train.en、train.zh、dev.en、dev.zh、test.en、test.zh。將它們複製到/tmp/nmt_zh/中。

訓練模型：
```
mkdir -p /tmp/nmt_model_zh
python -m nmt.nmt \
  --src=en --tgt=zh \
  --attention=scaled_luong \
  --vocab_prefix=/tmp/nmt_zh/vocab \
  --train_prefix=/tmp/nmt_zh/train \
  --dev_prefix=/tmp/nmt_zh/dev \
  --test_prefix=/tmp/nmt_zh/test \
  --out_dir=/tmp/nmt_model_zh \
  --step_per_stats 100 \
  --num_train_steps 200000 \
  --num_layers 3 \
  --num_units 256 \
  --dropout 0.2 \
  --metrics bleu
```

在/tmp/my_infer_file.en中儲存一些需要翻譯的英文句子（格式為：每一行一個英文句子，句子中每個英文單字，內含標點符號之間都要有
空格分隔。可以從test.en中複製）。使用下面的指令進行測試：

```
python -m nmt.nmt \
  --out_dir=/tmp/nmt_model_zh \
  --inference_input_file=/tmp/my_infer_file.en \
  --inference_output_file=/tmp/output_infer
```

翻譯後的結果被儲存在/tmp/output_infer檔案中。

#### 拓展閱讀

- 關於用Encoder-Decoder 結構做機器翻譯工作的更多細節，可以參考 原始論文Learning Phrase Representations using RNN Encoder– Decoder for Statistical Machine Translation。

- 關於注意力機制的更多細節，可以參考原始論文Neural Machine Translation by Jointly Learning to Align and Translate。此外還有改進 版的注意力機制：Effective Approaches to Attention-based Neural Machine Translation。
