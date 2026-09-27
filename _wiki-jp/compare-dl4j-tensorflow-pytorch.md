---
title: ディープラーニングの比較シート：TensorFlow、Keras、PyTorch、Caffe、MxNet、CNTK
short_title: ディープラーニングの比較シート：TensorFlow、Keras、PyTorch、Caffe、MxNet、CNTK
description: ディープラーニングの比較シート：TensorFlow、Keras、PyTorch、Caffe、MxNet、CNTK
---

### 目次

Lua

* <a href="#intro">Torch及びPytorch</a>

Pythonのフレームワーク

* <a href="#intro">Theanoとエコシステム</a>
* <a href="#tensorflow">TensorFlow</a>
* <a href="#caffe">Caffe</a>
* <a href="#chainer">Chainer</a>
* <a href="#cntk">CNTK</a>
* <a href="#dsstne">DSSTNE</a>
* <a href="#dsstne">DyNet</a>
* <a href="#keras">Keras</a>
* <a href="#mxnet">Mxnet</a>
* <a href="#paddle">Paddle</a>
* <a href="#bigdl">BigDL</a>
* <a href="#intro">ライセンス</a>

## Lua

### <a name="torch">Torch及びPytorch</a>

[**Torch**](http://torch.ch/)は、Luaで書かれたAPIを持つ計算フレームワークで、機械学習アルゴリズムをサポートします。そのいくつかのバージョンはFacebookやTwitterなどの大手のIT企業でも使用されており、それぞれ社内チームを設けてディープラーニングプラットフォームのカスタマイズに従事しています。Luaはマルチパラダイムのスクリプト言語で、1990年代初めにブラジルで開発されました。

Torch7はパワフルですが、Pythonを基盤とした学術的コミュニティーやJAVAを共通語とする企業ソフトエンジニアが[幅広くアクセスできることを目的として作成されたものではありませんでした](https://news.ycombinator.com/item?id=7929216)。

[Pytorch](https://github.com/pytorch/pytorch)として知られるTorchのPython APIは2017年1月にオープンソースになりました。PyTorchは様々な長さの入出力データを処理できる動的計算グラフを提供しています。これは例えば再帰型ニューラルネットワークを使用するときに役に立ちます。動的計算グラフをサポートしている他のフレームワークに、CMUのDyNet及びPFNのChainerがあります。

長所と短所：

* (+) 簡単に組み合わせることのできるモジュールのピースが多くある。
* (+) 自分の層のタイプを簡単に入力し、GPU上で実行できる。
* (+) Lua (^.^)（ライブラリのコードのほとんどがLuaにあり、読みやすい。）
* (+) 前もってトレーニングされたモデルが多くある。
* (+) PyTorch
* (-) Lua
* (-) 通常、自分でトレーニングのコードを入力する（プラグ・アンド・プレイがあまりない）。
* (-) 商業用サポートがない。
* (-) ドキュメンテーションが不十分

## Pythonのフレームワーク

### <a name="theano">Theanoとエコシステム</a>

ディープラーニングの分野における研究者たちは、ディープラーニングのフレームワークの祖父であり、[Python](http://darkf.github.io/posts/problems-i-have-with-python.html)によって書かれた[**Theano**](http://deeplearning.net/software/theano/)を使っています。TheanoはNumpyのように多次元配列を処理するライブラリです。他のライブラリと使用し、データ探索に適しており、研究を目的としています。

数多くのオープンソースのディープライブラリはTheanoの最上部に構築されています。これには[Keras](https://github.com/fchollet/keras)、[Lasagne](https://lasagne.readthedocs.org/en/latest/)、[Blocks](https://github.com/mila-udem/blocks)が含まれます。これらのライブラリは、常に直感的であるわけではないTheanoのインターフェイスの最上部に、使い方が簡単なAPIを配置させることを目的としています。（2016年3月以降、Theano関連のライブラリである[Pylearn2は使用不可となっているようです](https://github.com/lisa-lab/pylearn2))。

長所と短所：

* (+) Python + Numpy
* (+) 計算グラフは見やすく抽象化されている。
* (+) 再帰型ニューラルネットワークは計算グラフに良く適合する。
* (-) 生のTheanoは幾分レベルが低い。
* (+) 高レベルのラッパー（Keras、Lasagne）により、手間が省ける。
* (-) エラーメッセージが役に立たないことがある。
* (-) 大きなモデルの場合、コンパイルの時間が長くなることがある。
* (-) Torchよりも「分厚い」
* (-) 事前トレーニングを受けたモデルにへのサポートが不十分
* (-) AWSでのバグが多い。
* (-) 単一GPU

### <a name="tensorflow">TensorFlow</a>

* GoogleはTheanoの代替としてTensorFlowを作成しました。これら二つのライブラリは実際には非常に類似しています。Ian Goodfellow氏などのTheanoの作成者のうち何人かは、OpenAI社へと移る前にGoogle社にてTensorflowも作成しました。
* 現在のところ、**TensorFlow**は、いわゆる「インライン」の行列演算に対応しておらず、その上演算を実行するためには行列をコピーしなければなりません。非常に数の多い行列をコピーするのはあらゆる意味でコストがかかります。TensorFlowだと、最先端のディープラーニングツールの4倍の時間が掛かります。Googleはこの問題に取り組んでいる最中とのことです。
* ほとんどのディープラーニングのフレームワークのように、TensorFlowはC/C++エンジンの上にあるPython APIによって書かれているため、より速く動作します。Java APIには実験的サポートが提供されていますが、現在それは安定したものとは考えられておらず、Java及びScalaのコミュニティーにとっては解決策であるとは考えられていません。
* TensorFlowは、CNTKやMxNetなど[他のフレームワークと比べて動作は随分遅めです](https://arxiv.org/pdf/1608.07249v7.pdf)。
* TensorFlowは、ディープラーニング以外のことができます。TensorFlowは、強化学習やその他のアルゴリズムに対応するツールを備えています。
* Googleも自認しているその目標とは、人員を増やし、研究者のコードを共有可能なものとし、ソフトウェアエンジニアのディープラーニングへのアプローチ方法を標準化し、TensorFlowが最適化されたGoogleのクラウドサービスの人気を高めることです。
* TensorFlowは商用サポートを受けていません。Googleがオープンソースの企業ソフトウェアをサポートするビジネスを行うという可能性は低いでしょう。研究者に新しいツールを提供するものなのです。
* Theanoのように、TensforFlowは計算グラフ（例：行列演算の一連であるz = sigmoid(x)、xやzは行列）を生成し、自動識別を行います。自動識別は重要です。これにより、新しいニューラルネットワークの配列で実験しているたびにバックプロパゲーションの新しいバリエーションをハンドコーディングする必要がなくなるからです。Googleのエコシステムでは、計算グラフはGoogle Brainが最も難しい仕事を処理するのにも使用しますが、Googleはこれらのツールをオープンソース化していません。TensorFlowはGoogleの社内デイープラーニング・ソリューションの半分を占めています。
* ビジネスの観点から言うと、企業が考えるべきことは、これらのツールの提供をGoogleに求めたいかということです。
* Tensorflowの演算すべてがNumpyと同じようには動作しないのでご注意ください。

長所と短所：

* (+) Python + Numpy
* (+) Theanoと同じような計算グラフの抽象化
* (+) Theanoよりもコンパイル時間が短い。
* (+) TensorBoardを使用して視覚化する。
* (+) データとモデルの並列処理
* (-) 他のフレームワークより遅い。
* (-) Torchよりも「ぶ厚い」。魔法のようなことがもっとある。
* (-) 事前トレーニングされたモデルがあまりない。
* (-) 計算グラフは単なるPythonなため、遅い。
* (-) 商用サポート無し
* (-) 新しいトレーニングのバッチがあるごとに読み込みをするためにPythonにドロップアウトする。
* (-) ツールにはなりにくい。
* (-) 大規模なソフトウェアプロジェクトで動的型付けにエラーが発生しやくすい。

### <a name="caffe">Caffe</a>

[**Caffe**](http://caffe.berkeleyvision.org/)は有名で、広く一般に使用されているマシンビジョンのライブラリで、Matlabのの実装した素早い畳み込みネットワークをC及びC++で移植したものです（速度とテクニカル面の負担のトレードオフを考慮したい方は、あるチップから別のチップにC++を移植することについてのSteve Yegge氏の見解を[こちら](https://sites.google.com/site/steveyegge2/google-at-delphi)でお読みいただけます）。Caffeは、テキスト、音声、時系列データなど他のディープラーニングのアプリケーションを対象としたものではありません。これまでご紹介してまいりました他のフレームワークと同様に、CaffeはそのAPIが目的でPythonを選択しました。

長所と短所：

* (+) フィードフォワード（順伝播型）ネットワークと画像処理に適している。
* (+) 既存のネットワークの微調整に適している。
* (+) コードを書かずにモデルをトレーニングする。
* (+) Pythonのインターフェースが非常に有益である。
* (-) 新しいGPU層にC++/CUDAを書く必要がある。
* (-) 再帰型ニューラルネットワークには適していない。
* (-) 大規模なネットワークには（GoogLeNet, ResNet）使いにくい。
* (-) 拡張性がない。
* (-) 商業用サポートがない。
* (-) 使用されなくなりつつある可能性が高い。開発に時間が掛かる。

### <a name="cntk">CNTK</a>

[**CNTK**](https://github.com/Microsoft/CNTK)とは、マイクロソフトによるオープンソースのディープラーニングフレームワークで、CNTKとは「Computational Network Toolkit（計算ネットワークのツールキット）」の略語です。そのライブラリにはフィードフォワードディープニューラルネットワーク、畳み込みネットワーク、再帰型ニューラルネットワークネットが含まれています。そして、C++コードでPython APIを提供しています。CNTKは[permiLICENSEssive license（許容的ライセンス）](https://github.com/Microsoft/CNTK/blob/master/.md)を取得していると誤解されがちですが、ASF 2.0、BSD、MIT.など従来のどのライセンスも採用していません。許容的ライセンスはCNTKの1ビットSGD（確率的勾配降下法）といって、分散型トレーニングが簡単になる方法には適用されません。商業用には適用されないライセンスだからです。

### <a name="chainer">Chainer</a>

Chainerとはオープンソースのニューラルネットワークフレームワークで、Python APIを備えています。中心となる開発チームは[Preferred Networks](https://www.crunchbase.com/organization/preferred-networks#/entity)という東京を拠点とする機械学習のスタートアップで活動しており、多くの東京大学のエンジニアが携わっています。CMUのDyNetやFacebookのPyTorchが登場するまではChainerが動的計算グラフの主要なニューラルネットワークフレームワークであり、様々な長さの入力が可能なネットワークであったため、自然言語処理作業には人気の高い機能です。独自の[ベンチマーク](http://chainer.org/general/2017/02/08/Performance-of-Distributed-Deep-Learning-Using-ChainerMN.html)を使い、Chainerは他のPython向けフレームワークよりも素早く、MxNetとCNTKを含めた最も遅いテストグループであるTensorFlowを備えています。

### <a name="dsstne">DSSTNE</a>

Amazonの[DSSTNE](https://github.com/amznlabs/amazon-dsstne)（Deep Scalable Sparse Tensor Network Engine）は機械学習及びディープラーニングのモデル構築用のライブラリです。DSSTNEは、TensorflowやCNTKに次いで、近々リリースが予定されている多くのオープンソース・ディープラーニング・ライブラリのうちの一つです。AmazonはすでにAWS（Amazon Web Services）によってMxNetをサポートしているため、その将来は明確ではありません。DSSTNEはそのほとんどがC++コードで書かれており、他のライブラリほど人気は集めていませんが、作業速度は素早いようです。

* (+) スパースエンコーデイングに対応
* (-) Amazonは[最高結果を得るために必要なすべての情報やそのサンプル](https://github.com/amznlabs/amazon-dsstne/issues/24)をシェアしていない可能性がある。
* (-) Amazonは別のフレームワークを選択してAWSに使用している。

### <a name="dynet">DyNet</a>

[DyNet](https://github.com/clab/dynet)（[Dynamic Neural Network Toolkit](https://arxiv.org/abs/1701.03980)）はカーネギーメロン大学が開発し、元々CNNと呼ばれていました。その注目すべき機能は、動的計算グラフであり、様々な長さの入力が可能なため自然言語処理作業に適しています。PyTorch及びChainerも同じ機能を備えています。

* (+) 動的計算グラフ
* (-) ユーザコミュニティが小規模

### <a name="keras">Keras</a>

[Keras](https://keras.io)とは、Theano及びTensorFlowの最上部で使用するディープラーニングライブラリで、Torchを模範とした直感的なAPIを提供しています。現在利用可能なPython APIの中でも最も優れたものかもしれません。作成者はGoogleのソフトウェアエンジニアの[Francois Chollet氏](https://twitter.com/fchollet)です。

* (+) Torchを模範とした直感的なAPI
* (+) Theano、TensorFlow、Deeplearning4jのバックエンドに対応（CNTKバックエンドも今後追加される予定）
* (+) フレームワークが急速な成長を続けている。
* (+) ニューラルネットワークの標準的なPython APIとなる可能性が高い。

### <a name="mxnet">MxNet</a>

[MxNet](https://github.com/dmlc/mxnet)はAPIのある機械学習フレームワークで、言語はR、Python、Juliaなど[Amazonのウェブサービスが採用する](http://www.allthingsdistributed.com/2016/11/mxnet-default-framework-deep-learning-aws.html)言語を使うことができます。Apple社も2016年のGraphlab/Dato/Turi社買収後にApple製品の一部にこれを使用するとも言われています。MxNetは高速で柔軟性の高いライブラリで、Pedro Domingos氏及びワシントン大学の研究者チームがその開発に従事しています。MxNetとDeeplearning4jのいくつかの特徴の[比較分析](comparison-frameworks-dl4j-tensorflow-pytorch)をこちらでお読みいただけます。

### <a name="paddle">Paddle</a>

[Paddle](https://github.com/PaddlePaddle/Paddle)は、[Baidu氏が作成し、サポートする](http://www.infoworld.com/article/3114175/artificial-intelligence/baidu-open-sources-python-driven-machine-learning-framework.html)ディープラーニングフレームワークです。その正式名称はPArallel Distributed Deep LEarningです。Paddleはリリースが予定されているフレームワークのうち、最も主要なものです。その他のフレームワークと同様、Python APIを提供しています。

### <a name="bigdl">BigDL</a>

[BigDL](https://github.com/intel-analytics/BigDL)は新しいディープラーニングフレームワークで、Apache Sparkに主に重点を置いており、Intelチップのみで使用可能です。

### <a name="ml">機械学習のフレームワーク</a>

先に挙げたようなディープラーニングのフレームワークは一般的な機械学習フレームワークよりも専門化されたものです。現在、専門化は機械学習フレームは数多くあります。ここでは主なものだけを挙げましょう。

* [sci-kit learn](http://scikit-learn.org/stable/) - Python向けのデフォルトの機械学習フレームワーク
* [Apache Mahout](https://mahout.apache.org/users/basics/quickstart.html) - 最も重要なApacheの機械学習フレームワーク。Mahoutが分類、クラスタリング、リコメンデーションを行います。
* [SystemML](https://systemml.apache.org/) - IBMの機械学習フレームワークで、記述統計学、分類、クラスタリング、回帰、行列の因数分解、生存率解析を行う。サポートベクターマシンを含みます。
* [Microsoft DMTK](http://www.dmtk.io/) - Microsoftの分散型機械学習ツールキット分散型の語の埋め込みとLDA（線形判別分析）

### <a name="tutorial">Pathmindのチュートリアル</a>

* [ディープニューラルネットワークについて](neural-network)
* [畳込みネットワークのチュートリアル](convolutionalnets)
* [LSTMと回帰ネットワークのチュートリアル](lstm)
* [回帰ネットワークをDL4Jに使用しましょう](usingrnns)
* [ディープ・ビリーフ・ネットワークとMNIST](deepbeliefnetwork)
* [DataVecを使った Data Pipelineのカスタマイズ](image-data-pipeline)
* [制限付きボルツマン・マシン](restrictedboltzmannmachine)
* [固有ベクトル、PCA（主成分分析）、エントリピー](eigenvector.html)
* [Word2vec、Doc2vec、GloVe](word2vec)
