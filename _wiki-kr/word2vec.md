---
title: Word2vec - Java에서 인공 신경망으로 Word Embeddings 구현하기
short_title: Word2vec - Java에서 인공 신경망으로 Word Embeddings 구현하기
description: Word2vec - Java에서 인공 신경망으로 Word Embeddings 구현하기
---

내용

* <a href="#intro">소개</a>
* <a href="#embed">신경망과 Word Embeddings</a>
* <a href="#crazy">재미있는 Word2vec 결과</a>
* <a href="#just">예제 코드</a>
* <a href="#anatomy">DL4J에서 Word2vec 구조</a>
* <a href="#setup">준비, 데이터 로딩 및 학습</a>
* <a href="#code">코드 예제</a>
* <a href="#trouble">문제 해결 및 Word2Vec 튜닝하기</a>
* <a href="#use">Word2vec 이용 사례</a>
* <a href="#foreign">외국어</a>
* <a href="#glove">GloVe (Global Vectors) & Doc2Vec</a>

## <a name="intro">Word2Vec 소개</a>

Word2vec은 텍스트를 처리하는 인공 신경망이며 두 개의 층으로 구성되어 있습니다. Word2vec은 말뭉치(corpus)를 입력으로 받아서 말뭉치의 단어를 벡터로 표현 하는 방법을 찾는데, 이 벡터의 값은 말뭉치에서 단어가 가지는 의미나 역할을 잘 표현해주는 값이어야 합니다. 이렇게 단어의 의미와 맥락을 고려하여 단어를 벡터로 표현한 것을 word embeddings라고 합니다. Word2vec은 [심층 신경망](neural-network)은 아니지만 심층 신경망은 전처리 단계로 많이 쓰입니다.

Word2vec의 응용 분야는 매우 다양합니다. 가장 흔한 예는 텍스트로 된 문장을 이해하는 것 입니다. 그 외에도 word2vec의 구조는 <a href="#sequence">DNA 염기서열, 코드, 음악 재생목록, 소셜 미디어에서 사람들의 관계망 (graph)</a>를 이해하는데 사용합니다. 

Word2vec을 이용하면 단어간 유사성을 구할 수 있습니다. 원래 word embeddings의 목적이 유사한 단어일 수록 가까운 거리에 위치하도록 각 단어에 해당하는 벡터 값을 찾는 것 입니다. 이 학습은 사람이 관여하지 않으며 말뭉치 데이터만을 사용합니다.

데이터의 양이 충분하면 Word2vec은 단어의 의미를 꽤 정확하게 파악합니다. 그리고 이를 이용하면 단어의 뜻 뿐만 아니라 여러 단어의 관계를 알아냅니다. 예를 들어 단어의 관계를 이용해 '남자':'소년' = '여자':x 같은 관계식을 주면 x='소녀'라는 답을 구할 수 있습니다. 단어 뿐만 아니라 더 큰 단위의 텍스트인 문장이나 문서를 분류하는데에도 Word2vec을 사용합니다. 예를 들어 문서를 군집화한 뒤에 결과를 이용하면 검색 엔진에서 문서의 분야별 검색(과학, 법률, 경제 등)이나 문장의 감정 분석, 추천 시스템을 만들 수 있습니다.

정리하면, Word2vec은 각 단어마다 단어에 해당하는 벡터를 구해줍니다. 이 벡터를 다시 심층 신경망에 집어넣어서 추가적인 일을 할 수도 있고 단어의 유사성 등 관계를 파악할 수 있습니다.

유사성을 구하는 방법은 여러 가지가 있습니다. 흔히 쓰이는 방법은 [코사인 유사도](../glossary.html#cosine)입니다. 코사인 유사도는 두 벡터의 각도를 측정하는 것으로 각도가 같은 경우, 즉 두 벡터가 이루는 각이 0도인 경우엔 유사도의 최대값인 1.0이 나옵니다. 그리고 가장 유사도가 낮은 경우는 두 벡터의 각도가 90도가 되는 경우입니다 (실제로 90도가 나오는 경우는 거의 없습니다). 예를 들어 '스웨덴'과 '노르웨이'의 유사성을 구하면 0.760124 라는 제법 높은 유사도가 나올 것 입니다.

아래에 Word2vec을 이용해 구한 단어의 embeddings 중에서 '스웨덴'과 가장 거리가 가까운, 즉 가장 유사한 단어를 모아놓았습니다.

![Alt text](https://deeplearning4j.org/img/sweden_cosine_distance.png)

스칸디나비아 반도의 여러 국가와 기타 북유럽, 독일계 나라가 가장 가까운 단어 9개를 차지했습니다.

## <a name="embed">Neural Word Embeddings</a>

이렇게 뉴럴 네트워크를 이용해 word embeddings를 구하는 것을 *neural word embeddings*이라고 합니다.

즉, word embeddings은 사람의 언어를 컴퓨터의 언어로 번역하는 것 입니다.

단어를 같은 의미지만 다른 표현인 벡터로 바꿔준다는 점에서 Word2vec은 오토인코더와 비슷한 면이 있습니다. 하지만 [RBM(restricted Boltzmann machines)](restrictedboltzmannmachine)의 [재구성(reconstruction)](restrictedboltzmannmachine.html#reconstruct) 과정과 Word2vec의 학습 과정은 좀 다릅니다. Word2vec은 입력한 말뭉치의 문장에 있는 단어와 인접 단어의 관계를 이용해 단어의 의미를 학습합니다.

Word2vec의 학습 방법은 두 종류가 있습니다. CBOW(Continous Bag Of Words) 방식은 주변 단어가 만드는 맥락을 이용해 타겟 단어를 예측하는 것이고 skip-gram은 한 단어를 기준으로 주변에 올 수 있는 단어를 예측하는 것 입니다. 대규모 데이터셋에서는 skip-gram이 더 정확한 것으로 알려져있으며 저희도 이 방식을 이용합니다.

![Alt text](https://deeplearning4j.org/img/word2vec_diagrams.png)

Word2vec의 학습 과정은 큰 틀에서 일반적인 인공 신경망의 학습과 비슷합니다. 한 단어에 이미 할당된 벡터, 즉 word embedding이 있다고 가정하면 이 값을 이용해 주변 문맥을 얼마나 정확하게 예측하는지 계산합니다. 그리고 정확도가 좋지 못한 경우, 즉 추가적인 조정이 필요한 경우에 오차에 따라 벡터의 값을 업데이트합니다. 즉, 학습 과정에서 한 단어를 기준으로 단어 주변의 문맥을 참고하여 현재 embedding 벡터가 얼마나 정확한지, 오차의 값은 어느 정도인지를 알아냅니다. 만일 어떤 두 단어가 비슷한 문맥에서 꾸준하게 사용될 경우 두 단어의 벡터 값은 비슷하게 됩니다.

전체 차원이 500차원일 경우 이 embedding 벡터는 500차원 공간에 있는 점 하나에 해당합니다. 3차원 이상의 공간은 머릿속에서 상상하기 어렵지만 word embedding은 보통 수백차원의 공간을 사용합니다.

학습이 잘 완료되었다면 이 고차원 공간에서 비슷한 단어는 근처에 위치하게 됩니다. 예를 들어 나무의 종류인 *oak*, *elm* 및 *birch* 는 비슷한 곳에 모이게 됩니다. 또 의미에 유사성이 있는 *war*(전쟁), *conflict*(갈등) 및 *strife*(불화)는 다른 위치에 모이게 됩니다.

비슷한 물체나 개념은 가까이에 위치합니다. 그리고 단어의 상대적인 의미와 관계 또한 이 공간의 관계에 잘 변환됩니다. 이 관계를 이용하면 유사성을 넘어서 더 복잡한 일을 할 수 있습니다. 이를 테면 아래 예제와 같은 일 입니다.

![Alt text](https://deeplearning4j.org/img/word2vec_translation.png)

이 예제에서 두 가지를 이해하시면 됩니다. 로마, 파리, 베를린, 베이징은 전부 나라의 수도이며 비슷한 의미와 맥락에서 쓰이기 때문에 가까이 위치합니다. 뿐만 아니라, 벡터의 합과 차를 이용하면 재미있는 결과를 얻을 수 있습니다. 각 수도를 뜻하는 단어는 각 나라와 같은 관계에 있습니다. 즉 로마와 이탈리아, 베이징과 중국은 모두 수도와 국가의 관계이므로 각각에 로마-이탈리아와 베이징-중국의 벡터 공간에서의 관계도 유사하게 학습이 되어야 합니다. 실제로 Word2vec을 이용해 로마(의 embedding 벡터) - 이탈리아(의 embedding 벡터) + 중국(의 embedding 벡터)를 수행하면 베이징이 나옵니다. 왜냐하면 `로마-이탈리아`와 `중국-베이징`은 상대적으로 비슷한 벡터이기 때문입니다.

![Alt text](https://deeplearning4j.org/img/countries_capitals.png)

## <a name="crazy">재미있는 Word2Vec 사용 결과</a>

Word2vec을 이용한 다른 연산을 보겠습니다.

우선 더하기, 빼기, 등호 대신에 다른 기호를 사용하겠습니다. 수학에서 비례식은 1:2=5:10 으로 관계를 표현합니다. 이것과 유사하게 우리는 `:`와 `::`를 사용합니다. `::`은 등호(`=`)로 생각하시면 됩니다. 위의 예제에 적용하면, "로마에게 이탈리아가 있다면 베이징에겐?(정답은 중국)"의 표현을 로마:이탈리아::베이징:?? 으로 표현할 수 있습니다. 이렇게 하면 Word2vec이 적절한 단어를 골라 줍니다. 아래는 Word2vec이 고른 단어를 확률이 높은 순서대로 여러 개 나열했습니다.

```java
왕:여왕::남자:[여자, 유괴 미수, 10대, 여자 아이]
//조금 이상한 단어도 있지만 대체로 어느 정도 이해할 수 있습니다.

중국:대만::러시아:[우크라이나, 모스코바, 몰도바, 아르메니아]
//지정학적 및 외교적 관계를 반영한 결과가 나왔습니다. 모스코바는 조금 이상하지만요.

집:지붕::성:[돔, 종탑, 첨탑, 총탑, 포탑]

무릎:다리::팔꿈치:[팔, 팔뚝, 척골]

뉴욕타임즈::슐츠버그::폭스:[머독, 처닌, 뱅크로프트, 아일즈]
//슐츠버그는 뉴욕 타임즈의 소유주 및 경영자
//머독은 폭스 뉴스의 소유주
//피처 너닌은 폭스 뉴스의 최고업무책임자(COO)였음
//로저 아일즈는 폭스 뉴스의 회장
//뱅크로프트가는 월스트리트 저널을 머독에게 판매함

사랑:무관심::공포:[무관심, 냉담, 수줍음, 무력함, 무반응]

도날드 트럼프:공화당::버락 오바마:[민주당, 공화당, 민주당지지자, 매캐인]
//오바마와 매캐인의 라이벌 관계를 생각하면 Word2vec이 트럼프와 공화당의 관계를 적대적인 관계로도 해석한다고 볼 수 있습니다.

원숭이:사람::공룡:[화석, 화석화, 빙하기포유류]
//인류는 화석화된 원숭이다? 인류는 원숭이의 잔재다? 인류는 원숭이의 대결에서 승리한 경쟁자이다? 다 조금씩 말이 됩니다.

건물:건축가::소프트웨어:[프로그래머]
```

이 결과는 구글 뉴스 데이터셋을 사용해 학습한 것이며 이 데이터셋은 DL4J에서 언제든지 [import](#import)할 수 있습니다. 중요한 점은 Word2vec이 영어의 문법을 전혀 모르는 상태에서 이렇게 스스로 학습했다는 것 입니다. Word2vec은 아무런 언어 구조나 사전 없이 단시간에 엄청난 양의 단어를 학습합니다.

이번엔 다른 수식으로 Word2vec의 결과로 나온 embedding의 연산을 해봅시다. ([자세한 설명은 아래에서 다룹니다.](#eval))

* 지정학적 개념: *이라크 - 폭력 = 요르단*
* 구별, 차이점: *사람 - 동물 = 윤리*
* *대통령 - 권력 = 수상*
* *도서관 - 책 = 홀*
* 비유: *주식 시장 ≈ 온도계*

Word2vec에서 구한 유사 단어는 단어의 스펠링과 전혀 관계 없습니다. Word2vec은 단어의 의미를 기반으로 유사성을 구하며 결과적으로 단어를 고차원 공간 벡터를 이용해 나타냅니다.

### <a name="grams">N-grams & Skip-grams</a>

학습 과정에서 단어는 우선 하나의 벡터에 할당이 되고, *그 단어를 중심으로 전후의 몇 단어를 같이 읽습니다*. 이렇게 연속된 n개의 단어를 통째로 n-그램이라고 합니다. n-그램의 특수한 케이스로, 단어를 띄어쓰기 기준으로 하나의 단어를 하나의 개체로 보는 것을 유니그램(unigram, n=1), 두 개씩 이어서 생각하는 것을 바이그램(bigram, n=2)이라고 합니다. 즉 n-그램은 문서를 n개의 연속된 단어 단위로 자른 것 입니다. 예를 들어 문장이 '하나의 벡터에 할당이 된다.'라면 여기에는 4개의 유니그램 '하나의', '벡터에', '할당이', '된다'이 있는 것 입니다 (토큰화하는 과정은 생략하였습니다). 바이그램으로 표현하면 '하나의 벡터에', '벡터에 할당이' '할당이 된다' 이렇게 3개의 바이그램을 만들 수 있습니다. 스킵그램(skip-gram)은 n-그램에서 기준이 되는 단어를 제외한 것 입니다.

DL4J가 구현한 스킵그램은 Mikolov가 발표한 방법으로, CBOW보다 더 정확한 것으로 알려져있습니다.

말뭉치에서 추출한 n-그램을 Word2vec 신경망에 공급하면 단어의 벡터값을 찾아줍니다.



### <a name="patent">구글의 Word2vec 특허</a>

Word2vec는 Tomas Mikolov를 비롯한 구글의 연구자들이 출판한 논문 [단어의 벡터 표현들을 계산하는 방법](http://arxiv.org/pdf/1301.3781.pdf)을 통해 소개되었습니다. 구글은 Apache 2.0 라이센스를 적용한 [오픈 소스 버전의 Word2vec](https://code.google.com/p/word2vec/)를 호스팅하고 있습니다. 2014년, Mikolov는 구글을 떠나 페이스북으로 이직했고, 2015년 5월, [구글은 출시되어 온 Apache 라이센스를 폐지하지 않는 조건의 Word2vec 특허](http://patft.uspto.gov/netacgi/nph-Parser?Sect1=PTO2&Sect2=HITOFF&p=1&u=%2Fnetahtml%2FPTO%2Fsearch-bool.html&r=1&f=G&l=50&co1=AND&d=PTXT&s1=9037464&OS=9037464&RS=9037464)를 등록했습니다.

### <a name="foreign">외국어</a>

지금까지 살펴본 모든 과정은 언어에 관계 없이 적용 가능합니다. 하지만 적절한 데이터셋을 만들기 위한 자연어 전처리는 언어에 따라 다르게 적용되어야 합니다. 이 부분은 외부 라이브러리를 참고하시기 바랍니다. 예를 들어 [Stanford Natural Language Processing Group](http://nlp.stanford.edu/software/)은 [만다린 중국어](http://nlp.stanford.edu/projects/chinese-nlp.shtml), 아랍어, 프랑스어, 독일어 및 스페인어 등의 토큰화, 품사 태깅 및 고유명사 인식 등 다양한 기능을 갖고 있으며 Java로 구현되어 있습니다. 일본어는 [Kuromoji](http://www.atilika.org/)가 유명합니다. [텍스트 말뭉치 등 다국어 리소스는 여기](http://www-nlp.stanford.edu/links/statnlp.html)를 참고하시기 바랍니다.


### 학습 자료

* [DL4J의 Doc2vec(문서 벡터), 또는 단락 벡터](../doc2vec)
* [사고 벡터, 자연어 처리 & AI의 미래](../thoughtvectors)
* [Quora: Word2vec의 작동 원리](http://www.quora.com/How-does-word2vec-work)
* [Quora: Word2Vec을 이용한 재미있는 결과물](http://www.quora.com/Word2vec/What-are-some-interesting-Word2Vec-results/answer/Omer-Levy)
* [Word2Vec 소개](http://www.folgertkarsdorp.nl/word2vec-an-introduction/); Folgert Karsdorp
* [Mikolov'의 Word2vec 코드 원문 @구글](https://code.google.com/p/word2vec/)
* [word2vec 설명: Mikolov et al.’의 Negative-Sampling Word-Embedding 방식 도출하기](http://arxiv.org/pdf/1402.3722v1.pdf); Yoav Goldberg와 Omer Levy
* [Bag of Words & 용어 빈도-역 문서 빈도 (TF-IDF)](../bagofwords-tf-idf)

### 다른 초보자 가이드

* [심층 신경망 소개](neural-network)
* [초보자를 위한 RNNs과 LSTM 가이드](lstm)
* [고유 벡터(Eigenvectors), PCA, 공분산(Covariance) 및 엔트로피(Entropy)에 대한 기초 강의](eigenvector)
* [컨볼루션 네트워크](kr-convolutionalnets)
* [초보자용 RBM(Restricted Boltzmann Machines) 튜토리알](restrictedboltzmannmachine)
* [Neural Networks & Regression](../linear-regression)
* [심층학습(딥러닝) 활용 사례](use-cases)

### <a name="doctorow">문학 속의 Word2Vec</a>

수식은 마치 언어와 같다. 단어를 숫자로 번역하면 누구나 정확히 그 말을 이해할 수 있다. 목소리, 억양, 아, 어, 오 등 모든 발음이 사라지고 모든 오해가 해결되며 정확한 숫자로 생각을 포현한다. 모든 개념을 명확하게 표현하는 것이다.
    -- E.L. Doctorow, Billy Bathgate
