# Char-RNN-TensorFlow

Multi-language Char RNN in TensorFlow. You can use this code to generate English text, Chinese poetries and lyrics, Japanese text and text in other language.

一個基於最新版本TensorFlow的Char RNN實現。可以實現產生英文、寫詩、歌詞、小說、產生程式碼、產生日文等功能。


## Requirements
- Python 2.7.X
- TensorFlow >= 1.2

## Generate English Text

To train:

```
python train.py \
  --input_file data/shakespeare.txt  \
  --name shakespeare \
  --num_steps 50 \
  --num_seqs 32 \
  --learning_rate 0.01 \
  --max_steps 20000
```

To sample:

```
python sample.py \
  --converter_path model/shakespeare/converter.pkl \
  --checkpoint_path model/shakespeare/ \
  --max_length 1000
```

Result:

```
BROTON:
When thou art at to she we stood those to that hath
think they treaching heart to my horse, and as some trousting.

LAUNCE:
The formity so mistalied on his, thou hast she was
to her hears, what we shall be that say a soun man
Would the lord and all a fouls and too, the say,
That we destent and here with my peace.

PALINA:
Why, are the must thou art breath or thy saming,
I have sate it him with too to have me of
I the camples.

```

## Generate Chinese Poetries

To train:

```
python train.py \
  --use_embedding \
  --input_file data/poetry.txt \
  --name poetry \
  --learning_rate 0.005 \
  --num_steps 26 \
  --num_seqs 32 \
  --max_steps 10000
```

To sample:

```
python sample.py \
  --use_embedding \
  --converter_path model/poetry/converter.pkl \
  --checkpoint_path model/poetry/ \
  --max_length 300
```

Result:
```
何人無不見，此地自何如。
一夜山邊去，江山一夜歸。
山風春草色，秋水夜聲深。
何事同相見，應知舊子人。
何當不相見，何處見江邊。
一葉生雲裡，春風出竹堂。
何時有相訪，不得在君心。
```

## Generate Chinese Novels

To train (The file "novel.txt" is not included in this repo. You should find one and make sure it is utf-8 encoded!):
```
python train.py \
  --use_embedding True \
  --input_file data/novel.txt \
  --num_steps 80 \
  --name novel \
  --learning_rate 0.005 \
  --num_seqs 32 \
  --num_layers 3 \
  --embedding_size 256 \
  --lstm_size 256 \
  --max_steps 1000000
```

To sample:
```
python sample.py \
  --converter_path model/dpcq/converter.pkl \
  --checkpoint_path  model/novel \
  --use_embedding \
  --max_length 2000 \
  --num_layers 3 \
  --lstm_size 256 \
  --embedding_size 256
```

Result:
```
聞言，蕭炎一怔，旋即目光轉向一旁的那名灰袍青年，然後目光在那位老者身上掃過，那裡，一個巨大的石台上，有著一個巨大的巨坑，一些黑色光柱，正在從中，一道巨大的黑色巨蟒，一股極度恐怖的氣息，從天空上暴射而出 ，然後在其中一些一道道目光中，閃電般的出現在了那些人影，在那種靈魂之中，卻是有著許些強者的感覺，在他們面前，那一道道身影，卻是如同一道黑影一般，在那一道道目光中，在這片天地間，在那巨大的空間中，彌漫而開……

“這是一位斗尊階別，不過不管你，也不可能會出手，那些家伙，可以為了這裡，這裡也是能夠有著一些例外，而且他，也是不能將其他人給你的靈魂，所以，這些事，我也是不可能將這一個人的強者給吞天蟒，這般一次，我們的實力，便是能夠將之擊殺……”

“這裡的人，也是能夠與魂殿強者抗衡。”

蕭炎眼眸中也是掠過一抹驚駭，旋即一笑，旋即一聲冷喝，身後那些魂殿殿主便是對於蕭炎，一道冷喝的身體，在天空之上暴射而出，一股恐怖的勁氣，便是從天空傾灑而下。

“嗤！”
```

## Generate Chinese Lyrics

To train:

```
python train.py  \
  --input_file data/jay.txt \
  --num_steps 20 \
  --batch_size 32 \
  --name jay \
  --max_steps 5000 \
  --learning_rate 0.01 \
  --num_layers 3 \
  --use_embedding
```

To sample:

```
python sample.py --converter_path model/jay/converter.pkl \
  --checkpoint_path  model/jay  \
  --max_length 500  \
  --use_embedding \
  --num_layers 3 \
  --start_string 我知道
```

Result:
```
我知道
我的世界 一種解
我一直實現 語不是我
有什麼(客) 我只是一口
我想想我不來 你的微笑
我說 你我你的你
只能有我 一個夢的
我說的我的
我不能再想
我的愛的手 一點有美
我們 你的我 你不會再會愛不到
```

## Generate Linux Code

To train:

```
python train.py  \
  --input_file data/linux.txt \
  --num_steps 100 \
  --name linux \
  --learning_rate 0.01 \
  --num_seqs 32 \
  --max_steps 20000
```

To sample:

```
python sample.py \
  --converter_path model/linux/converter.pkl \
  --checkpoint_path  model/linux \
  --max_length 1000 
```

Result:

```
static int test_trace_task(struct rq *rq)
{
        read_user_cur_task(state);
        return trace_seq;
}

static int page_cpus(struct flags *str)
{
        int rc;
        struct rq *do_init;
};

/*
 * Core_trace_periods the time in is is that supsed,
 */
#endif

/*
 * Intendifint to state anded.
 */
int print_init(struct priority *rt)
{       /* Comment sighind if see task so and the sections */
        console(string, &can);
}
```

## Generate Japanese Text

To train:
```
python train.py  \
  --input_file data/jpn.txt \
  --num_steps 20 \
  --batch_size 32 \
  --name jpn \
  --max_steps 10000 \
  --learning_rate 0.01 \
  --use_embedding
```

To sample:
```
python sample.py \
  --converter_path model/jpn/converter.pkl \
  --checkpoint_path model/jpn \
  --max_length 1000 \
  --use_embedding
```

Result:
```
「ああ、それだ、」とお夏は、と夏のその、
「そうだっていると、お夏は、このお夏が、その時、
（あ、」
　と聲にはお夏が、これは、この膝の方を引寄って、お夏に、
「まあ。」と、その時のお庇《おも》ながら、
```

## Acknowledgement

Some codes are borrowed from [NELSONZHAO/zhihu/anna_lstm](https://github.com/NELSONZHAO/zhihu/tree/master/anna_lstm)

