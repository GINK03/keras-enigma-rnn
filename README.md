# KerasのRNNで暗号の一種であるシンプルなEnigmaを解きます  

## Enigma暗号とは
1918年に発明されたEnigmaは第二次世界大戦時に発明された暗号化機であり、電線の配線のパターンと、ロータといわれる入力するたびに回転する円盤のパターンで、様々な文字の置き換えを行います。

　ドイツ軍で用いたられたアルファベットの数だけ暗号化のもととなる配線が記された三つのロータを組み合わせて、膨大なパターンを構築して文字列を置換して、単純な交換則が成立しない複雑な暗号を構築して連合軍を苦しめました。
 
<div align="center">
  <img width="450px" src="https://user-images.githubusercontent.com/4949982/35311612-b7950b4a-00fa-11e8-803f-0c15835989a2.png">
</div>
<div align="center"> 図1. JavaScriptによるEnigma Simulator</div>

　連合国側のイギリスの特殊チームのULTRAによって解析されたようです。数学的な暗号の原理を追っているのですが、まだ完全にキャッチアップしきっておりませんが、群論とコンピュータのパワーとヒントとなるキーが人間の発想に依存するという特性を利用して解いたそうです。

2006年、分散コンピューティングを利用して総当たりにて、最後の未解決であったEnigma暗号を解いたとのことです[1]。

## どのようにEnigmaを連合国軍は解いたのか
二つの方法を組み合わせたような表現を確認しました。  

1. Bombeとよばれるコンピュータで総当たり攻撃した  
2. 人間が利用しがちなカギや文章（日付などが末尾に入るとか）から推定されるパターンを限定する  

<div align="center">
  <img width="200px" src="https://user-images.githubusercontent.com/4949982/35311427-cfb60c02-00f9-11e8-8e89-b0a0d672c13d.png">
</div>
<div align="center"> 図2. 現代に再現さたボンベ(Wikipediaより) </div>

## 今風の機械学習で解くにはどうすればいいのか
2.のように、何らかの暗号化前の文章と暗号化後の暗号文が手に入ったとします。このとき、この対が十分な量があるとき、ディープラーニングのアルゴリズムの一つであるRNNで解くことが可能です[2]  

ロータが絡み合うような、機械的な仕組みは、勝手にディープラーニングのような機械学習は苦手だろう思い込みがあったので、この資料を見たときは驚きました。

## ロータが二つのEnigmaを仮定
複雑な三つのロータではなく、簡単にした二つのロータのみで構成されるEnigmaを仮定します。 
Enigmaのロータは一文字進むごとに回転し、初期値が不明になっており、キーはランダムになっているとします.

初期値が不明なため、26(+3)^2パターンの成立しうるロータの状態をディープラーニングのネットワークを施行し、もっとも自然な文字列である初期状態とロータの配線を
全探索しないと、原理として解くことはできません。  

暗号化として以下のようなスクリプトを作成ました  

コーパスとしてBBCの公開ニュースコーパスを利用しています。  
```python
total = ''
for name in glob.glob('courpus/bbc/*/*'):
  try:
    text = open(name).read()
  except Exception as ex:
    continue
  buff = []
  for char in list(text.lower()):
    if char in char_index_0:
      buff.append(char)
  total += ''.join(buff)

# random slice
pairs = []
for index in random.sample( list(range(0, len(total) - 150)),100000):
  _char_index_0 = copy.copy(char_index_0)
  _char_index_1 = copy.copy(char_index_1)
  real = total[index:index+150]

  enigma = []
  for diff, char in enumerate(real):
    # roater No.1 update _char_index
    _char_index_0 = { char:(ind+1)%len(_char_index_0) for char, ind in _char_index_0.items() }
    # get index
    ind = _char_index_0[char]
    next_char = chars[ind]

    # roater No.2
    _char_index_1 = { char:(ind+1)%len(_char_index_1) for char, ind in _char_index_1.items() }
    # get index
    ind = _char_index_1[next_char]
    next_char = chars[ind]

    enigma.append(next_char)
  cript = ''.join(enigma)

  crop = random.choice(list(range(len(char_index_0))))
  real, cript = real[crop:crop+100], cript[crop:crop+100]
  pairs.append( (real, cript) )

open('pairs.json', 'w').write( json.dumps(pairs, indent=2) )
```

## DeepLearningのネットワークを設計
Kerasで実装しました。  
GRUを用いネットワークはこのようになっています。  
初期状態が不明なEnigmaで暗号化された暗号文を最大100文字入力し、対応する100文字を入力します  
```python
timesteps   = 100
inputs      = Input(shape=(timesteps, 29))
x           = Bi(GRU(512, dropout=0.10, recurrent_dropout=0.25, return_sequences=True))(inputs)
x           = TD(Dense(3000, activation='relu'))(x)
x           = Dropout(0.2)(x)
x           = TD(Dense(3000, activation='relu'))(x)
x           = Dropout(0.2)(x)
x           = TD(Dense(29, activation='softmax'))(x)

decript     = Model(inputs, x)
decript.compile(optimizer=Adam(), loss='categorical_crossentropy')
```

## 前処理
*Enigmaの配線をランダムで初期化*  
```console
$ python3 14-make-enigma.py 
```
*コーパスから暗号文と正解のペアを作成*  
```console
$ python3 15-prepare.py 
```
*RNNで入力する密ベクトルに変換*  
```console
$ python3 16-make-vector.py 
```

## 学習
全体の99%を学習します
```console
$ python3 17-train.py --train 
```

## 評価
学習で使わなかった1%のデータを評価します
```console
$ python3 17-train.py --predict
```

*出力*  
```console
[オリジナルの文] 　　 the emphasis on the islamic threat will just do the sameforsyth is wrong the nature of the current t    
[入力された暗号文] 　 dqyi q whqgsrtxutfnyqbjwivjferd ejmufyzreqwtwizzykscfzlxajemhkxogsrzovzvxwiqafxbbaxhjckjwdounmnjytwv
[モデルで評価した文]  the emphasis on the islamic threat will just do the sameforsyth is wrong the nature of the current t
```

このように、Enigmaのネットワークが未知であっても、確定してわかるテキストが十分にあれば、RNNでエニグマ暗号は解けることがわかりました。  
今回は仮想的なロータを一つソフトウェア的に再現しましたが、ロータが三つでも十分にRNNのネットワークが大きく、データが十分にあれば、この延長線上で解けると思います。

## 参考文献
- [1] [Wikipedia](https://ja.wikipedia.org/wiki/%E3%82%A8%E3%83%8B%E3%82%B0%E3%83%9E_(%E6%9A%97%E5%8F%B7%E6%A9%9F))
- [2] [Learning the Enigma with Recurrent Neural Networks](https://greydanus.github.io/2017/01/07/enigma-rnn/)
