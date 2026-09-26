---
title: Word2Vecとは？
short_title: Word2Vecとは？
description: Word2Vecとは？
---

目次

* <a href="#intro">Word2Vecとは？</a>
* <a href="#embed">ニューラル単語埋め込み</a>
* <a href="#crazy">面白いWord2Vecの出力結果</a>
* <a href="#use">Word2vecのユースケース</a>
* <a href="#foreign">外国語</a>

## <a name="intro">Word2Vecとは？</a>

Word2vecは、2層から成り、テキスト処理を行うニューラルネットワークです。  テキストコーパスを入力すると、出力結果には、ベクトルのセット、つまりコーパスにある単語の特徴量ベクトル（feature vector）が出されます。Word2vecは、[ディープ・ニューラル・ネットワーク](neural-network)ではありませんが、テキストをディープニューラルネットワークが理解できる数値形式に変えます。

Word2vecのアプリケーションが適用できる分野は、世間で使われる文の構文解析だけにとどまりません。パターンが識別される可能性のある<a href="#sequence">遺伝子、コード、再生リスト、ソーシャルメディアのグラフ、その他の文字列や記号列</a>にも適用できるのです。

Word2vecの目的及び有用性は、類似語のベクトルをベクトル空間にグループ化することです。つまり、数値に基づいて類似性を検知するのです。 Word2vecは、分散した単語の特徴（例えば個々の単語のコンテキストなど）の数値表現であるベクトルを作成します。これは人間の介在なしに行われます。

Word2vec は、十分なデータ、利用例、コンテキストが与えられれば、ある単語の意味の推測を、過去の出現例を基に、かなり高い精度で行うことが出来ます。これらの推測により、ある単語と他の単語との関連性を確立することができます（例えば、 「man（男性）」と「 boy（少年）」の関係は、 「woman（女性）」と「girl（少女）」の関係に当たるものである。）また、文書をクラスタリングし、トピックによって分類するのにも使用できます。検索やsentiment analysis（感情分析、レコメンデーションの際にこれらのクラスタをベースに使うことができます。その分野は、科学的研究、法的発見、電子商取引、顧客関係管理など、幅広く適用できます。

<!-- Word2vec は、十分なデータ、利用例、コンテキストが与えられれば、ある単語の意味の推測を、過去の出現例を基に、かなり高い精度で行うことが出来ます。これらの推測により、ある単語と他の単語との関連性を確立することができます（例えば、 「man（男性）」と「 boy（少年）」の関係は、 「woman（女性）」と「girl（少女）」の関係に当たるものである。）また、文書をクラスタリングし、トピックによって分類するのにも使用できます。検索や[sentiment analysis（感情分析）](../sentiment_analysis_word2vec)、レコメンデーションの際にこれらのクラスタをベースに使うことができます。その分野は、科学的研究、法的発見、電子商取引、顧客関係管理など、幅広く適用できます。 -->

Word2vecのニューラルネットワークで出力されるものは、単語の集合で、それぞれの単語にベクトルが付与されています。これらのベクトルはディープラーニング・ネットワークに入力したり、単語間の関係を検知するためにクエリーに使用することができます。

コサイン類似度を測定すると、90度としてで表される類似性はなく、完全一致を意味する類似度の1は、0度として表されます。スウェーデンはスウェーデンに等しく、ノルウェーは、スウェーデンとのコサイン距離がどの国よりも最も高く0.760124となっています。

以下は、Word2vecにより出力された「スウェーデン」と関連した単語のリストです。近接性の高いものから順に並べられています。

![Alt text](https://deeplearning4j.org/img/sweden_cosine_distance.png)

スカンジナビアの国々、いくつかの裕福な北ヨーロッパの国々、ゲルマン系の国々は上位9位に挙がっています。

## <a name="embed">ニューラル単語埋め込み</a>

単語を表現するために使用されるベクトルは、*neural word embeddings（ニューラル単語埋め込み）*と呼ばれていますが、その表現はちょっと変わっています。あるものが、別のあるものと根本的に異なるものであってもそれを表現します。かつて、Elvis Costelloが、言った「Writing about music is like dancing about architecture.（音楽について書くことは、建築について踊ることと似ている。）」という発言に似ています。 Word2vecは、単語を「vectorizes（ベクトル化）」し、コンピューターが自然言語を読み取れるようにします。単語の類似性を検知するパワフルな数学演算が実行できるようになったのです。

つまり、ニューラル単語埋め込みとは、数値の付与された語のことなのです。シンプルですが、翻訳とはまた異なります。

Word2vecは、各単語をベクトル化してエンコードするオートエンコーダーと似てはいますが、[制限付きボルツマン・マシン](restrictedboltzmannmachine)のように、入力された単語を[再構成](restrictedboltzmannmachine.html#reconstruct)させてトレーニングするのではなく、word2vecはある単語をその単語に近い単語との関連性に基づいてトレーニングします。

一般に単語のトレーニングには、2つの方法があります。コンテクストを使って対象語を推測する方法（continuous bag of words、CBOWと呼ばれる）、そして、ある語を使って、対象であるコンテクストを推測する方法です。後者の方法は、skip-gramと呼ばれますが、この方法を弊社は使用しています。こちらの方が、大きなデータセットでは、より精確な結果を生み出すからです。

![Alt text](https://deeplearning4j.org/img/word2vec_diagrams.png)

ある単語に割り当てられた特徴量ベクトルがその単語のコンテクストの精確な推測に使用できない場合、ベクトルのコンポーネントが調整されます。コーパスの中のそれぞれの単語が、*教師*のような役割を果たし、 特徴量ベクトルを調整させるためにエラーシグナルを送り返します。コンテクストからある単語とある単語のベクトルが類似していると判断された場合、ベクトルの数値を調整してより近くに移動させます。

例えば、ゴッホの絵画のひまわりが、カンバスに塗ったオイルの2次元混合で*表現する*1880年代後期におけるパリの3次元空間に存在した植物であるということに似ています。従って、ベクトルに配列された500の数値は、ある単語や単語の集団を表現することができるのです。

これらの数値は、500次元のベクトル空間にそれぞれの語を点に変えて配置していきます。3次元以上の空間は、視覚化することが困難です。（Geoff Hinton氏は、13次元の空間を想像してみてほしいと生徒に言ったことがありますが、生徒たちは、まず最初に3次元をイメージし、その後、13、13、13と何度も繰り返しつぶやくことしかできなかったと回想しています。）

十分にトレーニングされた語ベクトルのセットは、その空間に互いに似た語を配置します。単語の*oak*、*elm*、*birch*は、ある箇所に集まって配置され、*war*、*conflict*、*strife*は別の箇所に集まって配置されているという感じになるでしょう。

類似したものや考えは、「近い」ものとして表示されます。それらの相対的な意味が測定可能な距離として翻訳されます。質が量となり、アルゴリズムがその作業を行います。しかし、類似性は、Word2vecが学習できる多くの関連性の単なるベースに過ぎません。例えば、類似性は、ある言語におけるある単語とある単語の関係を測定し、別の言語にその関連性をマッピングすることもできるのです。

![Alt text](https://deeplearning4j.org/img/word2vec_translation.png)

これらのベクトルは、より包括的な単語の配列のベースに当たるものです。例えば、Rome、Paris、Berlin、Beijingは互いに近い距離に集まるでしょうが、これらの首都名はそれぞれ、それらの国名とも似た距離にあるでしょう。つまりRome - Italy = Beijing - Chinaとなるでしょう。そして、もし、ローマがイタリアの首都であるを知っていれば、中国の首都の名前を知らなくても、Rome -Italy + Chinaという式を使うと、Beijingという回答に行き着きます。これは決して冗談ではありません。

![Alt text](https://deeplearning4j.org/img/countries_capitals.png)

## <a name="crazy">面白いWord2Vecの出力結果</a>

Word2vecが産出する他の単語の関連性を見てみましょう。

Word2vecは、＋、－、＝などの記号を使わず、論理的類推に使用する表記法を使います。「:」は、「is to（対）」のようなもので、「::」は 「as（等しい）」のようなものです。つまり、 「Rome is to Italy as China is to Beijing（ローマとイタリアの関係は、中国と北京の関係に等しい。）」は、「Rome:Italy::Beijing:China」と表されます。 最後に、今度は「解答」1つをお見せするより、3つの要素を入れると、Word2vecが出力する単語のリストをお見せしましょう。

    king（王）:queen（女王）::man（男性）:[woman（女性）、Attempted abduction（誘拐未遂）、teenager（ティーンエイジャー）、girl（少女）]
    //ちょっと不思議なものも混じっていますが、だいたい何を指しているか分かります。

    China（中国）:Taiwan（台湾）::Russia（ロシア）:[Ukraine（ウクライナ）、Moscow（モスクワ）、Moldova（モルドバ）、Armenia（アルメニア）]
    //2つの大国と、少し疎遠になった小さな隣国。

    house（家）:roof（屋根）::castle（城）:[dome（ドーム）、 bell_tower（鐘楼）、 spire、crenellations、turrets（小塔）]

    knee（膝）:leg（脚）::elbow（肘）:[forearm（前腕）、 arm（腕）、ulna_bone（尺骨）]

    New York Times（ニューヨークタイムズ）:Sulzberger::Fox:[Murdoch, Chernin, Bancroft, Ailes]
    //The Sulzberger-Ochs家は、ニューヨークタイムズ社の所有者であり、経営者です。
    //The Murdoch家は、News Corp.の所有者であるため、Fox Newsを所有者です。
    //Peter Chernin氏は、13年間、News Corp.'s COO（最高執行責任者）でした。
    //Roger Ailes氏は、Fox Newsの社長です。
    //The Bancroft家が、 ウォールストリートジャーナルをNews Corpに売却しました。

    love（愛）:indifference（無関心）::fear（恐怖）:[apathy（無感動）, callousness（冷淡さ）, timidity（臆病さ）, helplessness（無力さ）, inaction（怠惰）]
    //この単語の配列はまるで詩のよう。素晴らしいです。

    Donald Trump（ドナルド・トランプ）:Republican（共和党）::Barack Obama（バラク・オバマ）:[Democratic（民主的な）、GOP（共和党のニックネーム）、 Democrats（民主党）、 McCain（マケイン）]
    //興味深いことに、オバマとマケインがライバル関係にあったように、
    //Word2vecは、トランプが共和党という概念とライバル関係にあると見なしているようです。

    monkey（猿）:human（人間）::dinosaur（恐竜）:[fossil（化石）、 fossilized（化石化した）、Ice_Age_mammals（氷河時代の哺乳類）、fossilization（化石化）]
    //人間は化石化した猿？残ったのが猿ではなく
    //人間? 人間は猿に打ち勝った種？
    //氷河時代の哺乳類が恐竜に打ち勝ったように? なるほど。

    building（建物）:architect（建築）::software（ソフトウェア）:[programmer（プログラマー）、SecurityCenter（セキュリティ・センター）、WinPcap]

このモデルのトレーニングは、Google News vocabを使って行いました。このGoogle News vocabは、[インポート](#import)して、遊ぶことができます。ちょっと考えてみてください。 Word2vecのアルゴリズムは、英語の文法などというものは何1つ教えてもらっていないないのです。そして世界のことなど何も知りません。規則に基づいた記号の論理や知識グラフなどとも無関係なのです。それでも、ほとんどの知識グラフが1年かけて人間の労働の力を借りながら学習するものよりも多くを、柔軟に、自動的に学習してしまうのです。Google Newsのドキュメントを使ったトレーニング前は白紙の状態ですが、トレーニング後には、人間にとって意味のある複雑な類推を計算することができるのです。

Word2vecモデルを使って別のタイプのクエリーも行うことができます。いつも鏡のように同じ関係を対応させた2つの類推でなくてもいいのです。 （[以下にその方法をご説明しましょう。](#eval)）

* Geopolitics（地政学）: *Iraq（イラク） - Violence（暴力） = Jordan（ヨルダン）*
* Distinction（区別）: *Human（人間） - Animal（動物） = Ethics（倫理）*
* *President （大統領）- Power（権限） = Prime Minister（首相）*
* *Library（図書館） - Books（本） = Hall（ホール）*
* Analogy（類推）: *Stock Market（株式市場） ≈ Thermometer（温度計）*

ある語と、他の類似語（必ずしも同じ文字を含まない文字で構成された）との近接性の認識を構築することにより、ある語の意味を固定したラベルのようなものから、より柔軟で一般的なものを表現できるようになりました。

### <a name="grams">N-gram & Skip-gram</a>

複数の語が同時にベクトルに読み込まれ、*特定の範囲内で前後に走査します*。これらの範囲は、n-gramといい、n-gramは、 ある言語テキストにある連続的な *n*の列です。単にユニグラム（unigram）、バイグラム（bigram）、トライグラム（trigram）、フォーグラム（four-gram）、ファイブグラム（five-gram）のn番目に当たるバージョンであるに過ぎません。skip-gramは、n-gramからアイテムをドロップさせるだけなのです。

skip-gram表現は、Mikolovによって普及され、DL4Jに実装されましたが、skip-gram表現を使うと、より一般的なコンテキストが生成されるため、CBOWなど他のモデルより精確だということが実証されています。

それから、このn-gramは、ニューラルネットワークに使われ、提供された語ベクトルの有意性を学習します。有意性とは、ある特定の大きな意味やラベルの指標（indicator）としての有用性として定義されています。

### <a name="patent">GoogleのWord2vec特許</a>

Word2vecは、[単語のベクトル表現を計算するメソッド](http://arxiv.org/pdf/1301.3781.pdf)であり、Tomas Mikolov氏の率いるGoogleの研究チームによって発表されました。Googleは、Apache 2.0ライセンスで提供された[Word2vecのオープンソース版をホスティング](https://code.google.com/p/word2vec/)します。2014年にMikolov氏は GoogleからFacebookに移り、2015年5月には[Googleは、そのメソッドで特許を取得](http://patft.uspto.gov/netacgi/nph-Parser?Sect1=PTO2&Sect2=HITOFF&p=1&u=%2Fnetahtml%2FPTO%2Fsearch-bool.html&r=1&f=G&l=50&co1=AND&d=PTXT&s1=9037464&OS=9037464&RS=9037464)しましたが、Apacheのライセンス自体を無効にするものではありません。

### <a name="foreign">外国語</a>

すべての言語の単語は、Word2vecを使ってベクトルに変換され、Deeplearning4jを使ってベクトルが学習されますが、自然言語処理にも各言語に固有な特徴があり、弊社のライブラリ以上のものが必要となる可能性があります。 [Stanford Natural Language Processing Group（スタンフォード大学自然言語処理グループ）](http://nlp.stanford.edu/software/)は、数多くのJavaベースのツールを開発し、トークン化、品詞のタグ付け、[標準中国語](http://nlp.stanford.edu/projects/chinese-nlp.shtml)、アラビア語、フランス語、ドイツ語、スペイン語などの固有表現抽出に使用されています。日本語の場合、[Kuromoji](http://www.atilika.org/)などの自然言語処理ツールが役に立ちます。テキストコーパスなど、その他の外国語のリソースは、[こちら](http://www-nlp.stanford.edu/links/statnlp.html)をお読みください。

### Doc2vec & その他のリソース

* [Doc2vec, or Paragraph Vectors, With Deeplearning4j（Deeplearning4jを使ったDoc2vec、またはパラグラフベクトル）](doc2vec)
* [Thought Vectors, Natural Language Processing & the Future of AI（思考ベクトル、自然言語処理、AIの未来）](thoughtvectors)
* [Quora: How Does Word2vec Work?（Quora：Word2vecの仕組みはどうなっているの？）](http://www.quora.com/How-does-word2vec-work)
* [Quora: What Are Some Interesting Word2Vec Results?（Quora：Word2Vecの面白い結果にはどんなものがあるの？）](http://www.quora.com/Word2vec/What-are-some-interesting-Word2Vec-results/answer/Omer-Levy)
* [Word2Vec: an introduction（Word2Vecの紹介）](http://www.folgertkarsdorp.nl/word2vec-an-introduction/); Folgert Karsdorp
* [Mikolov's Original Word2vec Code @Google（GoogleでのMikolovのオリジナルWord2vecコード）](https://code.google.com/p/word2vec/)
* [word2vec Explained: Deriving Mikolov et al.’s Negative-Sampling Word-Embedding Method（word2vecの説明：Mikolov et al.のネガティブサンプリング語埋め込みメソッド）](http://arxiv.org/pdf/1402.3722v1.pdf); Yoav Goldberg and Omer Levy
* [Bag of Words & Term Frequency-Inverse Document Frequency (TF-IDF)（CBOW＆単語の出現頻度）](bagofwords-tf-idf)

### <a name="doctorow">文学作品の中のWord2Vec</a>

    まるで数字が言語であるかのようなのだ。言語の文字それぞれが数字に取って代わり、言語と同じように、みんなそれを同じ意味に解釈する。文字の音も消失し、発音するときの、舌を打つとか、破裂音だとか、口蓋に舌が触れているかとか、オーだったか、アーだったかなどどいう調音方法や、誤った解釈がされる恐れのあるものすべて、音楽や映画が流す偽りのメッセージ、そんな世界は全部、その言葉の訛りと一緒に消えてしまって、全く新しい数字の言語というものが到来し、すべてが壁に書かれたもののように明瞭になる。だから、私が言ったように、数字による文章を読む時代がやって来たんだ。
        -- E.L. Doctorow著「Billy Bathgate（）」
