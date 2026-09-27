---
title: 프레임 워크 비교 - TensorFlow、Keras、PyTorch、Caffe、MxNet、및 CNTK
short_title: 프레임 워크 비교
description: 프레임 워크 비교 - TensorFlow、Keras、PyTorch、Caffe、MxNet、및 CNTK
---

### Content

Lua (루아)

* <a href="#torch">Torch & Pytorch</a>

파이썬 프레임워크

* <a href="#theano">Theano & 생태계</a>
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
* <a href="#licensing">라이센싱</a>

JVM 고려사항

* <a href="#speed">속도</a>
* <a href="#java">DL4J: 왜 JVM인가?</a>
* <a href="#ecosystem">DL4J: 생태계</a>
* <a href="#scala">DL4S: Scala기반 딥러닝</a>
* <a href="#ml">머신러닝 프레임워크</a>
* <a href="#tutorial">더 보기</a>

## Lua

### <a name="torch">Torch & Pytorch</a>

[**Torch**](http://torch.ch/)는 Lua언어로 작성된 API를 제공하는 계산 프레임 워크로 머신러닝 알고리즘을 지원합니다. Torch의 어떤 버전들은 페이스 북이나 트위터와 같은 대형 기술 회사에서 자체 버전을 개발하여 사용하고 있습니다. Lua는 다중 패러다임 스크립팅 언어로 1990 년대 초 브라질에서 개발되었습니다.

Torch7은 많은 장점이 있지만 파이썬 기반의 학계나 자바 언어를 사용하는 소프트웨어 엔지니어에게 [쉽게 접근할 수 있도록 설계되지는 않았습니다](https://news.ycombinator.com/item?id=7929216). 

[Pytorch](https://github.com/pytorch/pytorch)로 알려진 Torch용 Python API는 2017 년 1 월 Facebook에서 오픈 소스화되었습니다.  PyTorch는 가변 길이 입력 및 출력을 처리 할 수 있는 다이나믹 컴퓨테이션 그래프를 제공하며 특히 RNN을 사용할 때 유용합니다. 다이나믹 컴퓨테이션 그래프를 지원하는 다른 프레임 워크로는 CMU의 DyNet과 PFN 's Chainer가 있습니다.

장점과 단점:

* (+) 결합하기 쉬운 많은 모듈 조각
* (+) 자신만의 레이어 유형을 작성하고 GPU에서 실행하기 쉬움
* (+) Lua. ;) (대부분의 라이브러리 코드는 Lua로 되어있음, 읽기 쉬움)
* (+) 사전 학습된 모델 대거 존재
* (+) PyTorch
* (-) Lua
* (-) 보통 스스로 학습 코드를 작성해야 함 (적은 플러그 앤 플레이)
* (-) 상업적 지원 없음
* (-) 미완성 문서

## 파이썬 프레임워크

### <a name="theano">Theano와 생태계</a>

많은 딥러닝 분야의 학술 연구자들은 [파이썬](http://darkf.github.io/posts/problems-i-have-with-python.html)으로 작성된 딥러닝 프레임 워크의 대부로 불리는 [**Theano**](http://deeplearning.net/software/theano/)에 의존합니다.
Theano는 Numpy와 같은 다차원 배열을 다루는 라이브러리입니다.  Theano는 데이터 탐색에 적합하며 다른 라이브러리와 함께 연구용으로 개발되어 사용됩니다.

[Keras](https://github.com/fchollet/keras),  [Lasagne](https://lasagne.readthedocs.org/en/latest/), [Blocks](https://github.com/mila-udem/blocks) 등 수많은 오픈 소스 딥러닝 라이브러리가 Theano를 기반으로 탄생했습니다. 이 라이브러리들은 Theano의 일부 비직관적인 인터페이스 위에 API를 사용하기 쉽게 레이어를 추가하는 방식입니다. (2016 년 3 월 현재 [Pylearn2는 더이상 유효하지 않은 것](https://github.com/lisa-lab/pylearn2) 같습니다.)

장점과 단점

* (+) Python + Numpy
* (+) 컴퓨테이셔널 그래프에 적합한 추상화
* (+) 컴퓨테이셔널 그래프에 RNN이 잘 맞음.
* (-) Raw Theano는 수준이 낮은 편임.
* (+) 고급 래퍼 (Keras, Lasagne) 의 어려운 부분을 완화
* (-) 오류 메시지는 도움이 되지 않는 경우가 있음.
* (-) 대형 모델은 컴파일 시간이 오래 걸릴 수 있음.
* (-) 토치보다 훨씬 "복잡함"
* (-) 사전 학습 된 모델에 대한 패치 지원
* (-) AWS에서 잦은 버그가 발생
* (-) 단일 GPU

### <a name="tensorflow">TensorFlow</a>

* Google은 Theano를 대체하고자 TensorFlow를 만들었습니다. 사실 두 라이브러리는 아주 비슷합니다. Ian Goodfellow와 같은 Theano의 창시자 중 일부는 OpenAI로 떠나기 전에 Google에서 Tensorflow를 만들었습니다.
*	현재 **TensorFlow** 는 소위 "인라인" 행렬 연산을 지원하지 않지만 실행을 위해서는 행렬을 복사해야 합니다. 매우 큰 행렬을 복사하는 것은 모든 면에서 비용이 많이 듭니다. Tensorflow는 최신 딥러닝 도구에 비해 네 배나 시간이 더 걸립니다. 구글은 이 문제에 대해 연구하고 있다고 이야기 합니다.
* 대부분의 딥러닝 프레임 워크와 마찬가지로 TensorFlow는 C / C ++ 엔진에 Python API로 작성되어 빠른 실행이 가능합니다. Java 및 Scala  커뮤니티를 위한 것은 아닙니다.
* TensorFlow 는 CNTK와 같은 [다른 프레임워크 보다 속도면에서 많이 느립니다](https://arxiv.org/pdf/1608.07249v7.pdf).
* TensorFlow는 딥러닝 이상의 것입니다. TensorFlow에는 실제로 강화학습(Reinforcement Learning) 및 기타 알고리즘을 지원하는 도구가 있습니다.
* Tensorflow가 인정한 Google의 목표는 연구원들이 짠 코드를 공유하고 소프트웨어 엔지니어가 딥러닝에 접근하는 방법을 표준화하며 TensorFlow가 최적화 된 Google Cloud 서비스에 대한 추가적인 그리기를 창출하는 것입니다.
* TensorFlow는 상업적 지원이 되지 않으며 향후에도 Google은 엔터프라이즈를 위한 오픈 소스 소프트웨어 지원사업에 뛰어들 것 같지 않습니다. 현재는 그저 연구원들에게 툴을 제공할 뿐입니다.
* Theano와 마찬가지로 TensforFlow는 컴퓨테이션  그래프 (예 : z = sigmoid (x)와 같이 x와 z가 행렬인 일련의 행렬 연산)를 생성하고 자동으로 미분합니다. 자동 미분이 중요한 이유는 신경망을 업데이트 시키기 위한 역전파(Backpropagation) 변화량을 매번 수동으로 계산하고 반영할 필요가 없기 때문입니다. Google의 생태계에서 컴퓨테이션 그래프는 구글 브레인 (Google Brain)이중추적인 역할을 하였지만  아직 오픈 소스화 하지는 않았습니다. Google의 사내 딥러닝 솔루션의 절반은 Tensorflow로 구축되어 있습니다.
* 기업의 관점에서 볼 때, 생각해봐야 할 문제는 이러한 도구를 Google에 전적으로 의존하여 사용해야 하는가의 여부입니다.
*	주의 사항: Tensorflow의 모든 작업이 Numpy에서 하는 것처럼 작동하는 것은 아닙니다.

장점과 단점

* (+) Python + Numpy
* (+) 컴퓨테이션 그래프 앱스트랙션, Theano와 비슷
* (+) Theano보다 컴파일 시간이 빠름
* (+) TensorBoard 시각화
* (+) 데이터 및 모델 병렬 처리
* (-) 다른 프레임 워크보다 느림.
* (-) Torch보다 훨씬 “더 복잡”하지만 기능이 많음.
* (-) 사전 학습된 모델 부족.
* (-) 컴퓨테이션 그래프는 순수 파이썬으로 속도가 느림.
* (-) 상업적 지원 없음.
* (-) 파이썬으로 드롭 아웃되어 각각의 새로운 트레이닝 배치를 로드
* (-) 툴로 사용하기 어려움.
* (-) 다이나믹 타이핑은 대형 소프트웨어 프로젝트에서 오류가 발생하기 쉬움.

### <a name="caffe">Caffe</a>

[**Caffe**](http://caffe.berkeleyvision.org/)는 머신 비전 라이브러리로  Matlab이 C 및 C ++에 고속 컨볼루션 신경망을 구현한 것으로서 널리 사용되고 있습니다. (속도와 기술적인 빚(Technical Debt)의 트레이드오프를 알고 싶다면 [칩에서 칩으로 C ++ 포팅에 대한 Steve Yegge의 이야기](https://sites.google.com/site/steveyegge2/google-at-delphi))를 참조할 것). Caffe는 문자, 음성, 또는 시계열 데이터와 같은 다른 딥러닝 응용 프로그램을 위한 것이 아닙니다. 여기 언급된 다른 프레임 워크와 마찬가지로 Caffe는 API로 Python을 선택했습니다.


장점과 단점

* (+) 피드 포워드 네트워크 및 이미지 처리에 적합.
* (+) 기존 네트워크 미세 조정에 적합.
* (+) 코드 작성 없이 모델 트레이닝 가능.
* (+) 파이썬 인터페이스가 매우 유용.
* (-) 새로운 GPU 레이어에 C ++ / CUDA를 작성해야 함.
* (-) 회귀망에 적합하지 않음
* (-) 큰 네트워크를 다루기에는 불편함 (GoogLeNet, ResNet)
* (-) 확장성이 없음
* (-) 상업적 지원 없음
* (-) 개발이 느려지고 있으며, 곧 유효하지 않을 수 있음

### <a name="cntk">CNTK</a>

[**CNTK**](https://github.com/Microsoft/CNTK)는 Microsoft의 오픈 소스 딥러닝 프레임 워크입니다. Computational Network Toolkit"의 약자이며, 피드 포워드 DNN, 컨볼루션 넷 및 회귀망이 포함됩니다. CNTK는 C ++ 코드로 Python API를 제공합니다. CNTK는 [허용 라이센스](https://github.com/Microsoft/CNTK/blob/master/LICENSE.md)가있는 것처럼 보이지만 놀랍게도ASF 2.0, BSD 또는 MIT와 같은 일반적인 라이선스 중 하나를 채택하지는 않았습니다. 이 라이센스는 CNTK가 분산 교육을 쉽게 수행 할 수있는 방법 (1 bit SGD)에 적용되지 않으며, 상업적 용도로 부여되지도 않았습니다.

### <a name="chainer">Chainer</a>

CChainer는 Python API를 제공하는 오픈소스 신경망 프레임워크로, 개발자들 중 코어팀이 [Preferred Networks](https://www.crunchbase.com/organization/preferred-networks#/entity)에서 일하고 있습니다. Preferred Networks는 도쿄에 기반을 두고 있는 기계학습 스타트업이며 대부분의 엔지니어들이 도쿄대학 출신입니다.  Chainer는 CMU의 DyNet과 Facebook의 PyTorch가 출현하기 전까지 다이나믹 컴퓨테이션 그래프나 가변 길이의 입력을 지원하는 신경망으로 NLP 작업에 많이 사용되는 최고의 프레임워크였습니다. [벤치마크](http://chainer.org/general/2017/02/08/Performance-of-Distributed-Deep-Learning-Using-ChainerMN.html)에 따르면 Chainer는 다른 Python 기반 프레임워크보다 눈에 띄게 빠릅니다 TensorFlow는 MxNet 및 CNTK를 포함하는 테스트 그룹 중 가장 느린 속도를 보입니다.

### <a name="dsstne">DSSTNE</a>

Amazon의 Deep Scalable Sparse Tensor Network Engine ([DSSTNE](https://github.com/amznlabs/amazon-dsstne))은 머신러닝 및 딥러닝 모델 구축을 위한 라이브러리입니다. Tensorflow와 CNTK 이후 출시 된 많은 오픈 소스 딥러닝 라이브러리 중 가장 최근의 것이라고 볼 수 있습니다. Amazon은 AWS로 MxNet을 지원했기 때문에 미래가 명확하지 않지만, C ++로 작성된 DSSTNE는 속도 면에서 우수하다고 할 수 있습니다. 단, 다른 라이브러리만큼 큰 커뮤니티가 구축된 것은 아닙니다.

* (+) 스파스 인코딩 처리
* (-) Amazon은 최상의 결과를 얻는데 필요한 [모든 정보를 예제로 공유하지 않았을 수 있음](https://github.com/amznlabs/amazon-dsstne/issues/24).
* (-) Amazon은 AWS에서 사용할 다른 프레임 워크를 외부에서 선택함.

### <a name="dynet">DyNet</a>

[Dynamic Neural Network Toolkit](https://arxiv.org/abs/1701.03980) 인 [DyNet](https://github.com/clab/dynet)은 Carnegie Mellon University에서 나왔으며 cnn이라고 불려 왔습니다. 주목할만한 특징은 NLP에 적합한 가변 길이의 입력을 지원하는 다이나믹 컴퓨테이션 그래프입니다. PyTorch와 Chainer도 같은 것을 제공합니다

* (+) 다이나믹 컴퓨테이션 그래프
* (-) 작은 사용자 커뮤니티

### <a name="keras">Keras</a>

[Keras](https://keras.io)는 Theano와 TensorFlow를 백엔드(back-end)로 사용하는딥러닝 라이브러리로서 Torch에서 영감을 얻어 직관적 API를 제공합니다. 아마도 이것은 현존하는 최고의 파이썬 API 일 것입니다. 창시자는 Google의 소프트웨어 엔지니어인 [Francois Chollet](https://twitter.com/fchollet)입니다.


* (+) Torch에서 영감을 얻은 직관적 API
* (+) Theano와 작업 가능, TensorFlow와 Deeplearning4j 백엔드 (CNTK 백엔드 예정)
* (+) 빠르게 성장하고 있는 프레임워크
* (+) 신경망의 표준 Python API이 될 가능성이 큼

### <a name="mxnet">MxNet</a>

[MxNet](https://github.com/dmlc/mxnet)은 [Amazon Web Services](http://www.allthingsdistributed.com/2016/11/mxnet-default-framework-deep-learning-aws.html)에서 채택한 R, Python 및 Julia와 같은 언어를 API로 사용하는 머신러닝 프레임 워크입니다. 애플의 일부가 2016 년에 Graphlab / Dato / Turi가 인수된 후에도 사용하고 있다는 소문이 돌고 있습니다. MxNet은 빠르고 유연하며 현재 Pedro Domingos와 워싱턴 대학 연구원 팀이 참여하고 있습니다. MxNet과 Deeplearning4j의 장단점 [비교](comparison-frameworks-dl4j-tensorflow-pytorch)는 여기에서 확인할 수 있습니다.

### <a name="paddle">Paddle</a>

[Paddle](https://github.com/PaddlePaddle/Paddle)은 [Baidu가 만들고 지원](http://www.infoworld.com/article/3114175/artificial-intelligence/baidu-open-sources-python-driven-machine-learning-framework.html)한 딥러닝 프레임 워크로, “PArallel Distributed DeepLEarning”의 약자입니다. Paddle은 출시된 주요 프레임 워크 중 가장 최근의 것이며 다른 프레임 워크와 마찬가지로 Python API를 제공합니다.

### <a name="bigdl">BigDL</a>

[BigDL](https://github.com/intel-analytics/BigDL)은 Apache Spark에 초점을 맞춘 새로운 딥러닝 프레임 워크로서 Intel 칩에서만 작동합니다.

### <a name="ml">머신러닝 프레임워크</a>

위에 열거 된 딥러닝 프레임워크는 일반적인 머신러닝 프레임워크보다 심화되어 있습니다. 우리는 여기 주요 프레임 워크를 나열 할 것입니다 :

* [sci-kit learn](http://scikit-learn.org/stable/) - Python을 위한 기본 오픈 소스 머신러닝 프레임워크입니다.
* [Apache Mahout](https://mahout.apache.org/users/basics/quickstart.html) - Apache의 핵심 머신러닝 프레임워크입니다. Mahout은 분류, 클러스터링 및 권장 (recommendations)을 수행합니다.
* [SystemML](https://systemml.apache.org/) - IBM의 머신러닝 프레임워크로 기술 통계(Descriptive Statistics), 분류, 클러스터링, 회귀, 행렬 인수 분해 및 생존 분석을 수행하고 지원 벡터 시스템을 포함합니다.
* [Microsoft DMTK](http://www.dmtk.io/) - Microsoft의 분산 시스템 학습 툴 키트입니다. 분산 단어 임베딩 및 LDA.

### <a name="tutorial">Deeplearning4j Tutorials</a>

* [심층 신경망 소개](neural-network)
* [컴볼루셔널 네트워크 자습서](./convolutionalnets)
* [LSTM 및 회귀망 자습서](./lstm)
* [DL4J에서 회귀망 사용](./usingrnns)
* [MNIST를 사용한 심층 네트워크](./deepbeliefnetwork)
* [Canova를 사용하여 맞춤형 데이터 파이프라인 구축](./image-data-pipeline)
* [제한된 Boltzmann 기계](./restrictedboltzmannmachine)
* [고유벡터(Eigenvectors), PCA 및 엔트로피](./eigenvector.html)
* [Word2vec, Doc2vec & GloVe](./word2vec)
