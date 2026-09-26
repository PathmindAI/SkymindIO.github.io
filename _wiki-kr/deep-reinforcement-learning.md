---
title: 심층 강화 학습에 대한 초보자용 가이드
author: Chris V. Nicholson
short_title: 심층 강화 학습
description: 강화 학습이란 여러 단계를 거쳐 복잡한 목표를 달성하거나 특정 차원을 따라 최대화하는 방법을 학습하는 목표 지향적인 알고리즘이다.
---

*진실을 결정하는 것이 우리의 힘이 아니라면 , 우리는 가장 가능성이 높은 것에 따라 행동해야 한다. - Descartes*

목차

* <a href="#define">강화 학습(Reinforcement Learning)의 정의</a>
* <a href="#domain">강화 학습을 위한 도메인 선택</a>
* <a href="#reward">상태-행동 쌍과 보상의 복합 확률 분포</a>
* <a href="#time">머신 러닝과 시간의 관계</a>
* <a href="#neural">신경망과 심층 강화 학습(Deep Reinforcement Learning)</a>
* <a href="#code">코드 예제</a>
* <a href="#footnote">FootNotes</a>
* <a href="#reading">추가 참고 목록</a>

신경망은 최근 컴퓨터 비전, 기계 번역 및 시계열 예측과 같은 문제에서 눈부신 성과를 보여주고 있다. 또한 신경망과 강화 학습 알고리즘을 결합하여 [AlphaGo](https://deepmind.com/blog/alphago-zero-learning-scratch/){:target="_blank"}와 같은 놀라운 프로그램을 만들 수 있게 되었다.

강화 학습이란 여러 단계를 거쳐 복잡한 목표를 달성하거나 특정 차원을 따라 최대화하는 방법을 학습하는 목표 지향적인 알고리즘을 말한다. 예를 들면, 여러 가지 동작을 통하여 게임에서 얻는 점수를 최대화하는 것이다. 게임은 빈칸에서 시작하여, 적절한 조건 하에 초인적인 성과를 달성할 수 있다. 아이들이 잘못하면 벌을 받고 잘하면 칭찬으로 보상을 받는 것처럼, 이 알고리즘은 잘못된 결정을 내렸을 때 페널티를 받고 옳은 결정을 내렸을 때는 보상이 주어진다. 이것이 강화 학습이다.

딥 러닝을 결합한 강화 알고리즘은 [Atari 비디오 게임](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf) 전문가뿐만 아니라 바둑세계 챔피언도 능가할 수 있다. 이것을 대수롭지 않게 생각할 수도 있지만, 이전에 이루었던 업적에 비하면 엄청난 발전이며 기술의 발전은 급속도로 진전되고 있다.  

강화 학습은 즉각적인 행동과 그로 인해 생기는 지연된 보상(delayed return)과 관련된 어려운 문제를 해결한다. 강화 학습 알고리즘은 사람들이 그렇듯이 그들의 결정이 결실을 맺을 때까지 기다림의 시간이 필요하다. 예를 들어 바둑 게임에서의 보상 함수는 게임이 끝나고 계가(計家) 후에 주어진다. 즉 에이전트가 하는 행동의 결과가 먼 미래에 나타나는 지연 보상 환경에서는, 어떤 행동이 여러 단계를 거치며 어떤 결과를 초래하게 되는지 이해하기 어려울 수 있다.

강화 학습 알고리즘은 비디오 게임의 제한된 옵션에서 보다는, 어쩌면 더 모호한 환경인 현실에서 더욱 좋은 성능을 발휘할 것으로 기대된다. 사람들은 언젠가 이 알고리즘이 현실 세계의 목표를 달성하는 것에 큰 역할을 해내리라 믿고 있다.

[RL4J](https://github.com/deeplearning4j/rl4j){:target="_blank"}(Deeplearning4j library)에서 Deep-Q learning과 A3C 두 강화 학습 알고리즘을 확인해볼 수 있다. RL4J를 통해 [Doom 게임](https://www.youtube.com/watch?v=Pgktl6PWa-o){:target="_blank"}을 플레이할 수 있다.

<a class="w-button button-skilcta" href="https://skymind.ai/learn" style="width:75%; margin-top: 15px;" target="_blank">Learn to build AI apps now >></a>

## <a name="define">강화 학습(Reinforcement Learning)의 정의</a>

강화 학습은 에이전트(agent), 환경(environment), 상태(state), 행동(action), 보상(reward)의 개념을 사용하여 이해할 수 있다. 이 개념들은 아래에서 자세한 내용을 살펴볼 것이다. 각 개념들의 집합은 대문자로 표현하며, 이들의 특정 사례는 소문자로 표현한다. 예를 들어 `A`는 취할 수 있는 모든 행동들을 의미하고, `a`는 `A`에서의 특정한 행동을 의미한다.

* 에이전트: **에이전트**는 행동을 취한다. 여기서 에이전트란 배송 서비스를 수행하는 드론이나 비디오 게임에서 슈퍼 마리오를 예로 들 수 있다. 실세계에서는 바로 자기 자신이 에이전트라 할 수 있다.<sup>[1](#one)</sup>
* 행동 (A): `A`는 에이전트가 취할 수 있는 모든 **행동**을 말하며, 에이전트는 수행 가능한 행동의 리스트 중에서 앞으로 할 행동을 선택해야 한다. 비디오 게임에서의 리스트는 오른쪽이나 왼쪽으로 달리기, 높거나 낮게 점프하기, 웅크리거나 서있는 행동 등이 될 수 있다. 주식 시장에서의 리스트는 유가 증권 및 파생상품을 구매, 판매 또는 보유하는 것이 포함될 수 있다. 항공 드론을 다룰 때의 리스트에는 3D 공간에서의 여러 가지 속도와 가속도가 될 수 있다.
* 할인율(Discount factor): **할인율**은 보통 0과 1 사이의 값으로 설정되며 이 값에 시간이 지남에 따라 에이전트가 발견하는 미래의 보상이 곱해져 에이전트가 선택한 행동에 대한 보상 효과를 약화시킨다. 왜일까? 할인율은 즉각적으로 주어지는 보상보다 상대적으로 가치가 낮은 미래의 보상을 만들기 위해 고안되었다. 보통 그리스 문자인 감마(*γ*)로 표현되며 *γ*가 0.8이고 3단계를 거쳐 10점의 보상을 받는다면 보상의 현재가치는 `0.8³ x 10`이다. 1의 할인율은 즉각적인 보상만큼이나 가치 있는 미래의 보상을 만들 것이다. 이것은 [delayed gratification](https://en.wikipedia.org/wiki/Stanford_marshmallow_experiment) 실험에서 볼 수 있듯이, 사람들이 더 큰 만족을 위해 참고 기다리는 것과 같다.
* 환경(Environment): 에이전트가 이동하는 세계를 말한다. **환경**은 에이전트의 현재 상태 및 행동(입력)을 취하여 보상과 다음 상태(출력)를 반환한다. 당신이 에이전트라고 생각해보면 환경은 당신의 행동을 처리하고 그에 대한 결과를 결정하는 물리법칙과 사회의 규칙이 될 수 있다.
* 상태 (S): **상태(State)**란 에이전트가 인식하는 구체적이고 즉각적인 자신의 상황이다. 즉 도구, 장애물, 적, 또는 보상과 같은 중요한 요인들과 관련하여 에이전트가 마주하는 특정 장소와 시간이며 즉각적인 구성을 의미한다. 상태는 환경에 의해 반환되는 현재 또는 미래의 상황이라 할 수 있다.
* 보상 (R): **보상**이란 에이전트의 행동에 대한 성공이나 실패를 측정하는 피드백이다(ex. 슈퍼 마리오 게임에서 마리오가 동전에 닿으면 점수를 얻는 것). 특정 상태에서 에이전트는 선택한 행동을 출력 값으로 환경으로 보내고 환경은 그에 따른 에이전트의 새로운 상태(이전의 상태에서 취한 행동의 결과)와 보상을 반환한다. 에이전트의 행동에 의해 평가된 보상은 즉시 주어질 수도, 지연될 수 도 있다.
* 정책 (π): **정책**이란 에이전트가 현재 상태를 기준으로 다음의 행동을 결정하는 데 사용하는 전략이다. 에이전트는 특정한 상태에서 보상을 최대화할 수 있는 행동을 선택한다.
* 가치 (V): 단기적인 보상인 `R`과는 달리 `value`는 장기적인 관점에서의 할인된 기대값이다. `Vπ(s)`란 현재의 상태에서 정책 `π`에 따른 기대되는 보상을 의미한다. 여기에서는 보상을 할인하거나 미래의 기대값을 더욱 낮추기도 한다. 할인율의 개념을 생각해보자. “장기적으로 우리는 모두 죽는다”는 케인즈의 이론을 기억하자. 그것이 미래의 보상을 할인하는 이유이다.
* Q-value 또는 action-value (Q): Q-value와 Value의 비슷한데 **Q-value**는 현재의 상태에서 취하는 행동 a를 고려한다는 것이 차이점이다.  Qπ(s, a)은 정책 π에 따라 행동 a를 취할 경우 현재의 상태 s에서 받을 장기적인 리턴을 말한다. Q는 상태-행동 쌍을 보상에 매핑한다. 여기서 Q와 정책의 차이점에 주의해야 한다.
* 궤도(Trajectory): 그 상태에 영향을 미치는 연속적인 상태와 행동. 라틴어로 “~너머로 던지다(to throw across)”라는 뜻에서 파생된 단어이다. 에이전트의 삶은 시공간을 통해 높이 던져져 아치 형태를 이루는 공에 불과하다.

따라서 환경은 현재 상태에서 취한 행동을 다음 상태와 보상으로 바꾸는 함수이며, 에이전트는 새로운 상태와 보상을 다음 행동으로 변환하는 함수이다. 우리는 에이전트의 함수는 할 수 있지만 환경의 함수는 알 수 없다. 그것은 마치 블랙박스와 같이 입력값과 출력값만 볼 수 있다. 대부분의 사람들과 기술과의 관계처럼 우리는 그것이 무엇을 하는지 알지만 어떻게 작동하는지 알지 못한다. 강화 학습은 에이전트가 환경의 기능을 근사화하려는 시도를 나타낸다. 따라서 블랙박스 환경에 행동을 전송하여 보상을 최대화한다. 즉, 강화 학습 알고리즘은 에이전트가 앞으로 누적될 보상을 최대화하는 일련의 행동으로 정의되는 정책을 찾는 방법이다.


![simple RL schema](/images/wiki/simple_RL_schema.png){:target="_blank"}
*Credit: [Sutton & Barto](http://incompleteideas.net/book/bookdraft2017nov5.pdf)

위의 피드백 루프에서 첨자는 시간의 단계인 `t`와 `t+1`을 표현하고 각각은 다른 상태를 나타낸다. 즉 시간 `t`에서의 상태와  `t+1`에서의 상태를 나타낸다. 다른 형태의 머신러닝(지도 학습, 비지도 학습)과 다르게 강화 학습을 차례로 발생하는 상태-행동 쌍의 관점에서 순차적으로 생각해볼 수 있다.

강화 학습은 결과를 가지고 행동을 판단한다. 목표지향적이라고 할 수 있는데, 강화 학습의 목적은 에이전트의 목표를 달성하거나 목적 함수를 최대화하기 위한 연속적인 행동들을 학습하는 것이다.

다음과 같은 몇 가지 예를 살펴보자.

* 비디오 게임의 목표는 게임이 끝났을 때 가장 많은 점수를 얻어내는 것이다. 따라서 게임을 하면서 얻은 추가 포인트가 에이전트의 다음 행동에 영향을 미치게 된다. 즉 에이전트는 점수를 최대한 많이 얻기 위해 전함을 쏘고, 동전을 모으거나 떨어지는 유성을 피하는 방법 등을 학습하게 된다.
* 현실에서 로봇의 목표가 A지점에서 B지점으로 움직이는 것이라면, B지점에 더 가깝게 움직이는 로봇의 행동이 점수로 계산된다.

다음은 강화 학습의 목적 함수에 대한 예이다. 즉 목표를 정의하는 방식이라고 할 수 있다.

![reinforcement learning objective](/images/wiki/reinforcementlearning_objective.png){:target="_blank"}

시간을 의미하는 *t*에 보상 함수 *r*을 합산한다. 그래서 이 목적 함수는 게임을 실행하면서 얻는 모든 보상을 계산한다. 여기서 *x*는 주어진 시점에서의 상태이고 *a*는 그 상태에서 취한 행동이다. *r*은 *x*와 *a*에 대한 보상 함수이다. (여기서 현재 *감마 γ*는 고려하지 않는다.)

강화 학습은 입력값을 해석하는 방법에 따라 지도 학습 및 비지도 학습과 차이점이 있다. “무엇”을 학습하는지에 대해 설명하면서 그 차이점을 알아보도록 하자.

* 비지도 학습: 이 알고리즘은 레이블이 없이 유사점을 학습하고, 이를 확장하여 비정상적이거나 다른 점을 인식한 이상 탐지 등에 활용할 수 있다.
* 지도 학습: 이 알고리즘은 데이터 인스턴스와 레이블의 상관관계를 학습한다. 즉 레이블이 지정된 데이터 세트가 필요하다는 것인데 이 레이블은 알고리즘이 레이블을 예측할 때 잘못된 추측을 할 경우 알고리즘을 “지도”하고 수정하는 데 사용된다.
* 강화 학습: 단기, 장기적 보상에 기초한 행동으로 강화 학습을 피드백이 거의 없는 환경에서의 지도 학습으로 생각해볼 수 있다.

<iframe width="420" height="315" src="https://www.youtube.com/embed/-uXVu0l8guo" frameborder="0" allowfullscreen></iframe>

## <a name="domain">강화 학습을 위한 도메인 선택</a>

자율적인 강화 학습 에이전트는 한 맹인이 오직 귀와 지팡이로 세상을 탐색하는 것을 생각해볼 수 있다. 에이전트들은 각각의 작은 창문을 통해 주변 환경을 인식하지만 이 작은 창들이 그들의 상황을 인식하는 데 가장 적절한 방법이 아닐 수도 있다.

(사실 에이전트가 어떤 종류의 입력값과 피드백을 선택할지 결정하는 것은 어려운 문제이다. 이것을 도메인 선택이라고 한다. 비디오 게임하는 방법을 학습하는 알고리즘이 이 문제를 무시하는데 그 이유는 환경이 사람에 의해 만들어진 것이고 엄격하게 제한되기 때문이다. 따라서 비디오 게임은 강화 학습에 대한 아이디어를 테스트할 수 있는 최적의 환경을 제공한다. 도메인을 선택하기 위해서는 보통 문제 해결에 대한 지식이나 이론을 가진 인간의 결정이 필요하다. 예를 들면 자율주행 자동차 알고리즘의 도메인 입력값을 선택할 때 카메라와 GPS 데이터 외에 레이더 센서를 포함하도록 결정할 수 있다.)

![rat wired](/images/wiki/rat_wired.jpg)

## <a name="reward">상태-행동 쌍 및 보상의 복합 확률 분포</a>

강화 학습의 목적은 주어진 상태에서 최선의 행동을 선택하는 것인데, 이는 행동의 순위를 매기고 각 행동에 대한 상대적인 값을 부여한다는 것이다. 그러한 행동들은 상태에 따라 달라지기 때문에 실제로 측정하는 것은 상태-행동 쌍의 값이라 할 수 있다. 예를 들면 특정한 상태에서 취한 행동이라는 것은 즉 당신이 어딘가에서 했던 행동을 의미한다. 다음은 행동의 값과 의미가 상태에 따라 결정된다는 것을 보여주는 몇 가지 예제이다.

* 만약 그 행동이 누군가와 결혼하는 것이라고 가정해보자. 당신이 18살일 때 35살과 결혼하는 것은  당신이 90살일 때 35살과 결혼하는 것과는 아마 다른 의미일 것이다. 두 케이스는 다른 계기를 가지고 있을 것이고 서로 다른 결과를 가져올 것이다.

* 할리우드 영화에서 “Fire!”라고 외치는 장면이 나왔다고 생각해보자. 직접 그 장면을 보지 않고서는 이 말이 “불이야!”라고 외치는 것인지, 전쟁터에서 “쏴라”라고 명령하는 것인지 알 수 없다. 전후 맥락을 모르면 어떠한 행동의 결과를 예측하기 어렵다.

행동-상태 쌍을 앞서 설명한 Q 함수를 통해 기대되는 값으로 매핑한다. Q함수는 에이전트의 상태와 행동을 입력값으로 취하고 미래에 얻을 수 있는 보상에 매핑한다.

강화 학습이란 일련의 연속적인 상태-행동을 통해 에이전트가 행동을 수행하는 과정이며 행동의 결과로 얻는 보상을 관찰하여 에이전트가 취할 수 있는  최상의 경로를 정확하게 예측할 때까지 Q 함수를 보상에 적용하는 프로세스이다. 이 예측을 정책이라고 한다.

강화 학습은 매우 많은 수의 상태-행동 쌍과 관련된 보상의 복합 확률 분포를 모델링하려는 것이다. 이러한 이유로 강화 학습과 [마르코프 결정 과정](./markov-chain-monte-carlo)(복잡한 분포에서 샘플링하여 그 속성을 추론하는 방법)을 결합하여 사용한다. 이것은 [스타니스와프 울람이 몬테카를로 방법을 발명](http://permalink.lanl.gov/object/tr?what=info:lanl-repo/lareport/LA-UR-88-9068){:target="_blank"};하도록 영감을 준 문제, 즉 솔리테어 게임에서 승리할 확률을 추론하려고 하는 문제와 매우 흡사하다.

모든 통계적 접근을 본질적으로 파악하기란 쉽지 않다. 일부 현상(생물학, 정치학, 사회학 또는 보드 게임 관련)들은 매우 복잡하기 때문에 처음부터 이유를 밝혀내는 것이 불가능하다고 볼 수 있다. 이것을  연구하는 유일한 방법은 통계학을 통해서 표면적인 사건을 측정하고, 연관된 메커니즘이 잘 이해가 되지 않더라도 이들 사이의 상관관계를 밝혀내기 위해 시도하는 것이다. 강화 학습은 심층 신경망과 마찬가지로 데이터에서 정보를 추출하기 위해 샘플링에 의존하는 전략 중 하나이다.  

마르코프 결정 과정을 사용하여 상태-행동 쌍에 대한 보상의 확률 분포를 근사하고자 할 때, 강화 학습은 보상을 얻기 위한 행동만 반복하고 다른 방법은 시도하지 않을 수도 있다. 이미 알고 있는 방법을 고수(exploit)할 것인지 더 좋은 결과를 얻기 위해 새로운 행동을 지속적으로 탐색(explore)할지 결정하는 것은 강화 학습에서 매우 중요한 부분이다. 예를 들어, 석유 회사가 이미 보유한 유전에서 원유를 펌핑하는 동시에 새로운 유전을 시추하는 이중 기능을 하는 것처럼, 강화 학습 알고리즘은 당장의 즉각적인 보상으로 인해 미래의 더 좋은 결과를 놓치지 않기 위해 다양한 각도로 이용(exploitation)과 탐색(exploration) 둘 사이를 잘 조정해야 한다.

강화 학습은 반복적이다. 어떤 상태-행동 쌍을 취해야 어떤 보상을 얻어 낼 수 있는지 처음에는 알 수 없다. 운동선수나 음악가가 어떠한 상태에서 연습을 하여 퍼포먼스가 향상되는 것처럼 강화 학습도 반복적으로 특정 상태(state)에서 다양한 행동(action)을 시도하여 얻는 보상(reward) 통해 학습하는 것이다.

## <a name="time">머신 러닝과 시간의 관계</a>

알고리즘은 어떠한 문제를 빠르게 학습하는 방법이라 말할 수 있다. 강화 학습 알고리즘-시간의 관계는 인간-시간의 관계와 다르다. 알고리즘은 어느 상태에서 어떠한 행동을 하는 것이 최선인지 알 수 있을 때까지, 같은 상태에서 여러 가지 다른 동작을 동시에 실행할 수 있다.

또한 강화 학습 알고리즘은 인간보다 나은 조건에서 더 많이 학습할 수 있는 잠재력이 있다. 사실 알고리즘의 진정한 장점은 본질적인 특성보다는 동시에 수많은 실험이 가능하고 인간과는 달리 밤낮으로 피로감 없이 훈련할 수 있기 때문에 효율적인 학습이 가능하다는 것이다. AlphaGo와 같은 Go 게임에서 훈련된 알고리즘은 인간이 100년 동안 할 수 있는 것보다 더 많은 Go 게임을 수행할 것이다.<sup>[2](#two)</sup>

## <a name="neural">신경망과 심층 강화 학습(Deep Reinforcement Learning)</a>

신경망은 어디에 적합할까? 신경망은 상태-행동 쌍을 보상에 매핑하는 법을 학습하는 에이전트이다. 모든 신경망과 마찬가지로 입력을 출력과 관련시키는 함수를 근사하기 위해 상관 계수를 사용한다. 오류가 적어지는 gradient를 따라 weight를 반복적으로 조정하여 올바른 상관 계수 또는 weight를 찾기 위해 학습한다.  

강화 학습에서 컨볼루션 네트워크는 에이전트의 상태를 인식할 때  사용될 수 있다. (ex. 마리오가 플레이하는 화면, 드론이 떠 있는 지형) 즉, 이미지 인식이라는 전형적인 작업을 수행한다.

그러나 강화 학습의 컨볼루션 네트워크는 지도 학습과는 다르게 이미지를 해석한다. 지도 학습에서 네트워크는 이미지에 레이블을 적용한다. 즉 픽셀에 이름을 연결시키는 것이다.

![convolutional classifier](/images/wiki/conv_classifier.png)

실제로 지도 학습은 확률적인 관점에서 '이미지에 가장 잘 맞는 레이블의 순위'를 매긴다. 당나귀의 이미지를 보여주면 그 그림이 당나귀일 가능성 80%, 말이 될 가능성 50%, 개가 될 가능성 30%라고 결정할 수도 있다.

하지만 강화 학습에서 상태를 나타내는 이미지가 주어질 경우, 컨볼루션 네트워크가 해당 상태에서 수행할 수 있는 '행동의 순위'를 매긴다. 예를 들면 오른쪽으로 뛰면 5점, 점프는 7점, 왼쪽으로 뛰면 0의 보상을 얻을 것이라고 예측하는 것이다.

![convolutional agent](/images/wiki/conv_agent.png)

위의 이미지는 정책(policy) 에이전트가 상태를 최상의 행동으로 매핑하는 것을 보여준다.

![policy state action](/images/wiki/policy_state_action.png)

*상태를 행동으로 매핑하는 것을 정책(policy)이라고 한다.*

앞서 언급했던 것처럼 이것은 Q함수와는 다른데, Q는 상태-행동 쌍을 보상에 매핑하는 것이다.

좀 더 구체적으로 설명하자면, Q는 상태-행동 쌍을 궤도상의 이후에 취하는 행동으로부터 얻는 모든 미래의 보상과 즉각적인 보상의 가장 높은 조합으로 매핑한다. Q 방정식은 다음과 같다. (출처: 위키피디아)

![q learning equation](/images/wiki/q_learning_equation.png)

기대되는 보상값에 장기적으로 얻을 수 있는 할인된 기대값을 적용하면 Q함수는 단순히 가장 높은 Q-value를 가진 상태-행동 쌍을 선택한다.

강화 학습의 시작단계에서, 신경망의 상관계수는 확률적으로 또는 무작위로 초기화될 수 있다. 환경으로부터 얻는 피드백으로 신경망은 기대 보상과 ground-truth 보상의 차이를 사용하여 weight를 조정하고 상태-행동 쌍에 대한 해석을 향상시킬 수 있다.

이 피드백 루프는 지도 학습의 오류 역전파 알고리즘(backpropagation of error)과 유사하다. 그러나 지도 학습은 신경망이 예측하려고 하는 ground-truth 레이블이 무엇인지 아는 것으로 시작된다. 지도 학습의 목적은 서로 다른 이미지를 각각의 이름으로 매핑하는 모델을 만드는 것이다.

강화 학습은 새로운 행동들에 대한 반응으로 스칼라 숫자를 보내기 위해 환경에 의존한다. 환경에 의해 반환된 보상은 다양하거나 지연될 수 있고, 변수의 영향을 받아 피드백 루프에 노이즈를 유발할 수 있다.

이로 인해 Q 함수의 보다 완전한 표현이 가능해지는데, 이것은 어떠한 행동을 통하여 생성되는 즉각적인 보상뿐만 아니라 미래에 반환될 수 있는 지연된 보상도 고려한다.

## <a name="footnote">Footnotes</a>

<a name="one">1)</a> *강화 학습 알고리즘이 작동하는 모습을 상상하여 시각적으로 그려보는 것이 도움이 될 수 있다. 알고리즘이 슈퍼 마리오 게임을 배우는 중이라고 해보자. 이 알고리즘은 마리오가 게임을 하면서 가장 많은 점수를 획득하는 것을 목표로 한다. 그러기 위해서 우리는 수많은 마리오를 가능한 모든 게임 공간에서 실행시켜볼 수 있다. 1000개의 마리오가  산에 터널을 뚫고 , 땅을 파내는 것(ex. 게임 환경에 영향을 주는 행동을 반복해서 결정할 때)을 계속하면서 그들의 경험(ex. 터널을 뚫는 것)은 나무의 가지처럼 다양한 방식으로 뻗어 나간다. 마리오의 경험(ex. 터널)은 산을 가르는 빛의 통로이다. 인생과 마찬가지로, 하나의 성공적인 행동으로 인해 더욱 중요한 의사 결정을 할 경우, 성공적인 행동을 할 가능성이 더욱 높아지기 때문에 결국 마리오가 승리에 이르게 되는 것이다.
여기서 마리오가 에이전트라고 가정하고, 그의 상태-행동 쌍과 연관되는 보상을 추적하는 열 지도(각각의 상태-행동 쌍이 노란색에서 빨간색의 열로 표시되는 화면)가 있다고 상상해보자. 월 스트리트의 금융 거래인들이 보는 것처럼  마리오가 플레이하는 화면이 수많은 모니터에 격자로 조립되어 있다. 한 화면에서는 "이 상태에서 더 세게 점프"하는 행동을 취하고, 다른 화면에서는 "이 상태에서 더 빨리 실행"하는 행동을 취할 수 있다. 몇몇 상태-행동 쌍들은 다른 것들보다 훨씬 더 많은 보상을 이끌어내고 점프, 쪼그려 앉기, 달리기 같은 다른 종류의 행동을 취할 수 있기 때문에 행동에 대한 보상의 확률 분포는 belle curve가 아니라 더욱 복잡한 형태가 될 것이다. 따라서 [마르코프와 몬테 카를로 기법](./markov-chain-monte-carlo)이 이것을 탐색하기 위해 사용되는 것이다. 즉, 보상의 분포를 공식으로 기술하기는 어렵지만 샘플링할 수는 있다. 알고리즘은 아무것도 모르는 상태에서 시작했고 게임(상태 공간)을 통해 많은 경로들을 탐색한 것이 아니기 때문에 열지도는 그러한 경험의 부족을 반영할 것이다. 따라서 기대되는 보상을 나타내는 열지도에 공백이 생기거나 경험을 통해 조정되는 보상에 대한 디폴트를 가정하는 것으로 시작할 수도 있다. 마리오는 이러한 열지도에 따라 보상을 얻으려고 하고, 게임을 더 많이 실행할수록 미래의 잠재적인 보상에 대한 열지도가 더욱 정확해진다 .열지도는 기본적으로 마리오의 현재 상태에서 가능한 상태-행동 쌍에 대한 보상의 확률 분포라고 할 수 있다.*

<a name="two">2)</a> *정확히 비유하자면 실제로 학습 알고리즘을 하나의 종(species)으로 볼 수 있다. 알고리즘이 학습하는 대로 실행되는 각 시뮬레이션은 그 종의 각 개체로 간주될 수 있다. 알고리즘이 게임을 통해 쌓은 지식이 알고리즘 모델의 세계에서 수집되는 것처럼, 인간도 언어를 통해 이야기가 전해져 내려오고 텍스트, 기록, 구전으로 구현되어 더욱 지능적일 수 있게 된다. 이것은 인간이 세계 챔피언일 때조차도 알고리즘과 개별 인간 사이의 경연이 왜 불공평한지에 대해 더 자세히 지적한다.*

## <a name="reading">추가 참고 목록</a>

* Richard S. Sutton and Andrew G. Barto's [Reinforcement Learning: An Introduction](http://incompleteideas.net/sutton/book/the-book-2nd.html){:target="_blank"}
* [Reinforcement Learning in Java](https://github.com/deeplearning4j/deeplearning4j/tree/master/rl4j)
* Andrej Karpathy's [ConvNetJS Deep Q Learning Demo](https://cs.stanford.edu/people/karpathy/convnetjs/demo/rldemo.html){:target="_blank"}
* [Brown-UMBC Reinforcement Learning and Planning (BURLAP)](http://burlap.cs.brown.edu/)(Apache 2.0 Licensed as of June 2016)
* [Glossary of Terms in Reinforcement Learning](http://www-anw.cs.umass.edu/rlr/terms.html){:target="_blank"}
* [Reinforcement Learning and DQN, learning to play from pixels](https://rubenfiszel.github.io/posts/rl4j/2016-08-24-Reinforcement-Learning-and-DQN.html){:target="_blank"}
* Video: [Richard Sutton on Temporal Difference Learning](https://www.youtube.com/watch?v=EeMCEQa85tw){:target="_blank"}
* [A Brief Survey of Deep Reinforcement Learning](https://arxiv.org/pdf/1708.05866.pdf){:target="_blank"}
* [The Policy of Truth, by Ben Recht](http://www.argmin.net/2018/02/20/reinforce/){:target="_blank"}
* [Deep Reinforcement Learning Doesn't Work Yet](https://www.alexirpan.com/2018/02/14/rl-hard.html)
* [Machine Learning for Humans: Reinforcement Learning](https://medium.com/machine-learning-for-humans/reinforcement-learning-6eacf258b265){:target="_blank"}

## RL Theory

### Lectures
 - [UCL] [COMPM050/COMPGI13 Reinforcement Learning](http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching.html){:target="_blank"} by David Silver
 - [UC Berkeley] CS188 Artificial Intelligence by Pieter Abbeel
   - [Lecture 8: Markov Decision Processes 1](https://www.youtube.com/watch?v=i0o-ui1N35U){:target="_blank"}
   - [Lecture 9: Markov Decision Processes 2](https://www.youtube.com/watch?v=Csiiv6WGzKM){:target="_blank"}
   - [Lecture 10: Reinforcement Learning 1](https://www.youtube.com/watch?v=ifma8G7LegE){:target="_blank"}
   - [Lecture 11: Reinforcement Learning 2](https://www.youtube.com/watch?v=Si1_YTw960c){:target="_blank"}
 - [Udacity (Georgia Tech.)] [CS7642 Reinforcement Learning](https://classroom.udacity.com/courses/ud600){:target="_blank"}
 - [Stanford] [CS229 Machine Learning - Lecture 16: Reinforcement Learning](https://www.youtube.com/watch?v=RtxI449ZjSc&feature=relmfu){:target="_blank"} by Andrew Ng
 - [UC Berkeley] [Deep RL Bootcamp](https://sites.google.com/view/deep-rl-bootcamp/lectures){:target="_blank"}
 - [UC Berkeley] [CS294 Deep Reinforcement Learning](http://rll.berkeley.edu/deeprlcourse/){:target="_blank"} by John Schulman and Pieter Abbeel
 - [CMU] [10703: Deep Reinforcement Learning and Control, Spring 2017](https://katefvision.github.io/){:target="_blank"}
 - [MIT] [6.S094: Deep Learning for Self-Driving Cars](http://selfdrivingcars.mit.edu/){:target="_blank"}
   - [Lecture 2: Deep Reinforcement Learning for Motion Planning](https://www.youtube.com/watch?v=QDzM8r3WgBw&list=PLrAXtmErZgOeiKm4sgNOknGvNjby9efdf){:target="_blank"}


### Reinforcement Learning Books
 - Richard Sutton and Andrew Barto, Reinforcement Learning: An Introduction (1st Edition, 1998) [[Book]](http://incompleteideas.net/book/ebook/the-book.html) [[Code]](http://incompleteideas.net/book/code/code.html)
 - Richard Sutton and Andrew Barto, Reinforcement Learning: An Introduction (2nd Edition, in progress, 2018) [[Book]](http://incompleteideas.net/book/bookdraft2018jan1.pdf) [[Code]](https://github.com/ShangtongZhang/reinforcement-learning-an-introduction)
 - Csaba Szepesvari, Algorithms for Reinforcement Learning [[Book]](http://www.ualberta.ca/~szepesva/papers/RLAlgsInMDPs.pdf)
 - David Poole and Alan Mackworth, Artificial Intelligence: Foundations of Computational Agents [[Book Chapter]](http://artint.info/html/ArtInt_262.html)
 - Dimitri P. Bertsekas and John N. Tsitsiklis, Neuro-Dynamic Programming [[Book (Amazon)]](http://www.amazon.com/Neuro-Dynamic-Programming-Optimization-Neural-Computation/dp/1886529108/ref=sr_1_3?s=books&ie=UTF8&qid=1442461075&sr=1-3&refinements=p_27%3AJohn+N.+Tsitsiklis+Dimitri+P.+Bertsekas) [[Summary]](http://www.mit.edu/~dimitrib/NDP_Encycl.pdf)
 - Mykel J. Kochenderfer, Decision Making Under Uncertainty: Theory and Application [[Book (Amazon)]](http://www.amazon.com/Decision-Making-Under-Uncertainty-Application/dp/0262029251/ref=sr_1_1?ie=UTF8&qid=1441126550&sr=8-1&keywords=kochenderfer&pebp=1441126551594&perid=1Y6RG2EGRD26659CJHH9)

### Survey Papers
 - Leslie Pack Kaelbling, Michael L. Littman, Andrew W. Moore, Reinforcement Learning: A Survey, JAIR, 1996. [[Paper]](https://www.jair.org/media/301/live-301-1562-jair.pdf)
 - S. S. Keerthi and B. Ravindran, A Tutorial Survey of Reinforcement Learning, Sadhana, 1994. [[Paper]](http://www.cse.iitm.ac.in/~ravi/papers/keerthi.rl-survey.pdf)
 - Matthew E. Taylor, Peter Stone, Transfer Learning for Reinforcement Learning Domains: A Survey, JMLR, 2009. [[Paper]](http://machinelearning.wustl.edu/mlpapers/paper_files/jmlr10_taylor09a.pdf)
 - Jens Kober, J. Andrew Bagnell, Jan Peters, Reinforcement Learning in Robotics, A Survey, IJRR, 2013. [[Paper]](http://www.ias.tu-darmstadt.de/uploads/Publications/Kober_IJRR_2013.pdf)
 - Michael L. Littman, "Reinforcement learning improves behaviour from evaluative feedback." Nature 521.7553 (2015): 445-451. [[Paper]](http://www.nature.com/nature/journal/v521/n7553/full/nature14540.html)
 - Marc P. Deisenroth, Gerhard Neumann, Jan Peter, A Survey on Policy Search for Robotics, Foundations and Trends in Robotics, 2014. [[Book]](https://spiral.imperial.ac.uk:8443/bitstream/10044/1/12051/7/fnt_corrected_2014-8-22.pdf)

### Reinforcement Learning Papers / Thesis
Foundational Papers
 - Marvin Minsky, Steps toward Artificial Intelligence, Proceedings of the IRE, 1961. [[Paper]](http://staffweb.worc.ac.uk/DrC/Courses%202010-11/Comp%203104/Tutor%20Inputs/Session%209%20Prep/Reading%20material/Minsky60steps.pdf) (discusses issues in RL such as the "credit assignment problem")
 - Ian H. Witten, An Adaptive Optimal Controller for Discrete-Time Markov Environments, Information and Control, 1977. [[Paper]](http://www.cs.waikato.ac.nz/~ihw/papers/77-IHW-AdaptiveController.pdf) (earliest publication on temporal-difference (TD) learning rule)

**Reinforcement Learning Methods**
 - Dynamic Programming (DP):
   - Christopher J. C. H. Watkins, Learning from Delayed Rewards, Ph.D. Thesis, Cambridge University, 1989. [[Thesis]](https://www.cs.rhul.ac.uk/home/chrisw/new_thesis.pdf)
 - Monte Carlo:
   - Andrew Barto, Michael Duff, Monte Carlo Inversion and Reinforcement Learning, NIPS, 1994. [[Paper]](http://papers.nips.cc/paper/865-monte-carlo-matrix-inversion-and-reinforcement-learning.pdf)
   - Satinder P. Singh, Richard S. Sutton, Reinforcement Learning with Replacing Eligibility Traces, Machine Learning, 1996. [[Paper]](http://www-all.cs.umass.edu/pubs/1995_96/singh_s_ML96.pdf)
 - Temporal-Difference:
   - Richard S. Sutton, Learning to predict by the methods of temporal differences. Machine Learning 3: 9-44, 1988. [[Paper]](http://incompleteideas.net/papers/sutton-88-with-erratum.pdf)
 - Q-Learning (Off-policy TD algorithm):
   - Chris Watkins, Learning from Delayed Rewards, Cambridge, 1989. [[Thesis]](http://www.cs.rhul.ac.uk/home/chrisw/thesis.html)
 - Sarsa (On-policy TD algorithm):
   - G.A. Rummery, M. Niranjan, On-line Q-learning using connectionist systems, Technical Report, Cambridge Univ., 1994. [[Report]](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=3&ved=0CDIQFjACahUKEwj2lMm5wZDIAhUHkg0KHa6kAVM&url=ftp%3A%2F%2Fmi.eng.cam.ac.uk%2Fpub%2Freports%2Fauto-pdf%2Frummery_tr166.pdf&usg=AFQjCNHz6IrgcaaO5lzC7t8oEIBY9epozg&sig2=sa-emPme1m5Jav7YmaXsNQ&cad=rja)
   - Richard S. Sutton, Generalization in Reinforcement Learning: Successful examples using sparse coding, NIPS, 1996. [[Paper]](http://webdocs.cs.ualberta.ca/~sutton/papers/sutton-96.pdf)
 - R-Learning (learning of relative values)
   - Andrew Schwartz, A Reinforcement Learning Method for Maximizing Undiscounted Rewards, ICML, 1993. [[Paper-Google Scholar]](https://scholar.google.com/scholar?q=reinforcement+learning+method+for+maximizing+undiscounted+rewards&hl=en&as_sdt=0&as_vis=1&oi=scholart&sa=X&ved=0CBsQgQMwAGoVChMIho6p_MOQyAIVwh0eCh3XWAwM)
 - Function Approximation methods (Least-Square Temporal Difference, Least-Square Policy Iteration)
   - Steven J. Bradtke, Andrew G. Barto, Linear Least-Squares Algorithms for Temporal Difference Learning, Machine Learning, 1996. [[Paper]](http://www-anw.cs.umass.edu/pubs/1995_96/bradtke_b_ML96.pdf)
   - Michail G. Lagoudakis, Ronald Parr, Model-Free Least Squares Policy Iteration, NIPS, 2001. [[Paper]](http://www.cs.duke.edu/research/AI/LSPI/nips01.pdf) [[Code]](http://www.cs.duke.edu/research/AI/LSPI/)
 - Policy Search / Policy Gradient
   - Richard Sutton, David McAllester, Satinder Singh, Yishay Mansour, Policy Gradient Methods for Reinforcement Learning with Function Approximation, NIPS, 1999. [[Paper]](http://papers.nips.cc/paper/1713-policy-gradient-methods-for-reinforcement-learning-with-function-approximation.pdf)
   - Jan Peters, Sethu Vijayakumar, Stefan Schaal, Natural Actor-Critic, ECML, 2005. [[Paper]](https://homes.cs.washington.edu/~todorov/courses/amath579/reading/NaturalActorCritic.pdf)
   - Jens Kober, Jan Peters, Policy Search for Motor Primitives in Robotics, NIPS, 2009. [[Paper]](http://papers.nips.cc/paper/3545-policy-search-for-motor-primitives-in-robotics.pdf)
   - Jan Peters, Katharina Mulling, Yasemin Altun, Relative Entropy Policy Search, AAAI, 2010. [[Paper]](http://www.kyb.tue.mpg.de/fileadmin/user_upload/files/publications/attachments/AAAI-2010-Peters_6439%5b0%5d.pdf)
   - Freek Stulp, Olivier Sigaud, Path Integral Policy Improvement with Covariance Matrix Adaptation, ICML, 2012. [[Paper]](http://arxiv.org/pdf/1206.4621v1.pdf)
   - Nate Kohl, Peter Stone, Policy Gradient Reinforcement Learning for Fast Quadrupedal Locomotion, ICRA, 2004. [[Paper]](http://www.cs.utexas.edu/~pstone/Papers/bib2html-links/icra04.pdf)
   - Marc Deisenroth, Carl Rasmussen, PILCO: A Model-Based and Data-Efficient Approach to Policy Search, ICML, 2011. [[Paper]](http://mlg.eng.cam.ac.uk/pub/pdf/DeiRas11.pdf)
   - Scott Kuindersma, Roderic Grupen, Andrew Barto, Learning Dynamic Arm Motions for Postural Recovery, Humanoids, 2011. [[Paper]](http://www-all.cs.umass.edu/pubs/2011/kuindersma_g_b_11.pdf)
   - Konstantinos Chatzilygeroudis, Roberto Rama, Rituraj Kaushik, Dorian Goepp, Vassilis Vassiliades, Jean-Baptiste Mouret, Black-Box Data-efficient Policy Search for Robotics, IROS, 2017. [[Paper](https://arxiv.org/abs/1703.07261)]
 - Hierarchical RL
   - Richard Sutton, Doina Precup, Satinder Singh, Between MDPs and Semi-MDPs: A Framework for Temporal Abstraction in Reinforcement Learning, Artificial Intelligence, 1999. [[Paper]](http://www-anw.cs.umass.edu/~barto/courses/cs687/Sutton-Precup-Singh-AIJ99.pdf)
   - George Konidaris, Andrew Barto, Building Portable Options: Skill Transfer in Reinforcement Learning, IJCAI, 2007. [[Paper]](http://www-anw.cs.umass.edu/pubs/2007/konidaris_b_IJCAI07.pdf)
 - Deep Learning + Reinforcement Learning (A sample of recent works on DL+RL)
   - V. Mnih, et. al., Human-level Control through Deep Reinforcement Learning, Nature, 2015. [[Paper]](http://www.readcube.com/articles/10.1038%2Fnature14236?shared_access_token=Lo_2hFdW4MuqEcF3CVBZm9RgN0jAjWel9jnR3ZoTv0P5kedCCNjz3FJ2FhQCgXkApOr3ZSsJAldp-tw3IWgTseRnLpAc9xQq-vTA2Z5Ji9lg16_WvCy4SaOgpK5XXA6ecqo8d8J7l4EJsdjwai53GqKt-7JuioG0r3iV67MQIro74l6IxvmcVNKBgOwiMGi8U0izJStLpmQp6Vmi_8Lw_A%3D%3D)
   - Xiaoxiao Guo, Satinder Singh, Honglak Lee, Richard Lewis, Xiaoshi Wang, Deep Learning for Real-Time Atari Game Play Using Offline Monte-Carlo Tree Search Planning, NIPS, 2014. [[Paper]](http://papers.nips.cc/paper/5421-deep-learning-for-real-time-atari-game-play-using-offline-monte-carlo-tree-search-planning.pdf)
   - Sergey Levine, Chelsea Finn, Trevor Darrel, Pieter Abbeel, End-to-End Training of Deep Visuomotor Policies. ArXiv, 16 Oct 2015. [[ArXiv]](http://arxiv.org/pdf/1504.00702v3.pdf)
   - Tom Schaul, John Quan, Ioannis Antonoglou, David Silver, Prioritized Experience Replay, ArXiv, 18 Nov 2015. [[ArXiv]](http://arxiv.org/pdf/1511.05952v2.pdf)
   - Hado van Hasselt, Arthur Guez, David Silver, Deep Reinforcement Learning with Double Q-Learning, ArXiv, 22 Sep 2015. [[ArXiv]](http://arxiv.org/abs/1509.06461)
   - Volodymyr Mnih, Adrià Puigdomènech Badia, Mehdi Mirza, Alex Graves, Timothy P. Lillicrap, Tim Harley, David Silver, Koray Kavukcuoglu, Asynchronous Methods for Deep Reinforcement Learning, ArXiv, 4 Feb 2016. [[ArXiv]](https://arxiv.org/abs/1602.01783)
   - Simon Schmitt, Jonathan J. Hudson, Augustin Zidek, Simon Osindero, Carl Doersch, Wojciech M. Czarnecki, Joel Z. Leibo, Heinrich Kuttler, Andrew Zisserman, Karen Simonyan, S. M. Ali Eslami, Kickstarting Deep Reinforcement Learning, ArXiv, 10 Mar 2018, [Paper](https://arxiv.org/abs/1803.03835)

## Reinforcement Learning Applications

### Game Playing with Reinforcement Learning

Traditional Games
  - Backgammon - "TD-Gammon" game play using TD(λ) (Tesauro, ACM 1995) [[Paper]](http://www.bkgm.com/articles/tesauro/tdl.html)
  - Chess - "KnightCap" program using TD(λ) (Baxter, arXiv 1999) [[arXiv]](http://arxiv.org/pdf/cs/9901002v1.pdf)
  - Chess - Giraffe: Using deep reinforcement learning to play chess (Lai, arXiv 2015) [[arXiv]](http://arxiv.org/pdf/1509.01549v2.pdf)

Computer Games
  - Human-level Control through Deep Reinforcement Learning (Mnih, Nature 2015) [[Paper]](http://www.readcube.com/articles/10.1038%2Fnature14236?shared_access_token=Lo_2hFdW4MuqEcF3CVBZm9RgN0jAjWel9jnR3ZoTv0P5kedCCNjz3FJ2FhQCgXkApOr3ZSsJAldp-tw3IWgTseRnLpAc9xQq-vTA2Z5Ji9lg16_WvCy4SaOgpK5XXA6ecqo8d8J7l4EJsdjwai53GqKt-7JuioG0r3iV67MQIro74l6IxvmcVNKBgOwiMGi8U0izJStLpmQp6Vmi_8Lw_A%3D%3D) [[Code]](https://sites.google.com/a/deepmind.com/dqn/) [[Video]](https://www.youtube.com/watch?v=iqXKQf2BOSE)
  - [Flappy Bird Reinforcement Learning](https://github.com/SarvagyaVaish/FlappyBirdRL) [[Video]](https://www.youtube.com/watch?v=xM62SpKAZHU)
  - MarI/O - learning to play Mario with evolutionary reinforcement learning using artificial neural networks (Stanley, Evolutionary Computation 2002) [[Paper]](http://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf) [[Video]](https://www.youtube.com/watch?v=qv6UVOQ0F44)
  - [Montezuma's Revenge: Reinforcement Learning with Prediction-Based Rewards](https://blog.openai.com/reinforcement-learning-with-prediction-based-rewards/) from OpenAI

### Robotics with Reinforcement Learning

  - Policy Gradient Reinforcement Learning for Fast Quadrupedal Locomotion (Kohl, ICRA 2004) [[Paper]](http://www.cs.utexas.edu/~pstone/Papers/bib2html-links/icra04.pdf)
  - Robot Motor SKill Coordination with EM-based Reinforcement Learning (Kormushev, IROS 2010) [[Paper]](http://kormushev.com/papers/Kormushev-IROS2010.pdf) [[Video]](https://www.youtube.com/watch?v=W_gxLKSsSIE)
  - Generalized Model Learning for Reinforcement Learning on a Humanoid Robot (Hester, ICRA 2010) [[Paper]](https://ccc.inaoep.mx/~mdprl/documentos/Hester_2010.pdf) [[Video]](https://www.youtube.com/watch?v=mRpX9DFCdwI&list=PL5nBAYUyJTrM48dViibyi68urttMlUv7e&index=12)
  - Autonomous Skill Acquisition on a Mobile Manipulator (Konidaris, AAAI 2011) [[Paper]](http://lis.csail.mit.edu/pubs/konidaris-aaai11b.pdf) [[Video]](https://www.youtube.com/watch?v=yUICAkSQTZY)
  - PILCO: A Model-Based and Data-Efficient Approach to Policy Search (Deisenroth, ICML 2011) [[Paper]](http://mlg.eng.cam.ac.uk/pub/pdf/DeiRas11.pdf)
  - Incremental Semantically Grounded Learning from Demonstration (Niekum, RSS 2013) [[Paper]](http://people.cs.umass.edu/~sniekum/pubs/NiekumRSS2013.pdf)
  - Efficient Reinforcement Learning for Robots using Informative Simulated Priors (Cutler, ICRA 2015) [[Paper]](http://markjcutler.com/papers/Cutler15_ICRA.pdf) [[Video]](https://www.youtube.com/watch?v=kKClFx6l1HY)
  - Robots that can adapt like animals (Cully, Nature 2015) [[Paper](https://arxiv.org/abs/1407.3501)] [[Video](https://www.youtube.com/watch?v=T-c17RKh3uE)] [[Code](https://github.com/resibots/cully_2015_nature)]
  - Black-Box Data-efficient Policy Search for Robotics (Chatzilygeroudis, IROS 2017) [[Paper](https://arxiv.org/abs/1703.07261)] [[Video](https://www.youtube.com/watch?v=kTEyYiIFGPM)] [[Code](https://github.com/resibots/blackdrops)]


### Control with Reinforcement Learning

  - An Application of Reinforcement Learning to Aerobatic Helicopter Flight (Abbeel, NIPS 2006) [[Paper]](http://heli.stanford.edu/papers/nips06-aerobatichelicopter.pdf) [[Video]](https://www.youtube.com/watch?v=VCdxqn0fcnE)
  - Autonomous helicopter control using Reinforcement Learning Policy Search Methods (Bagnell, ICRA 2001) [[Paper]](http://repository.cmu.edu/cgi/viewcontent.cgi?article=1082&context=robotics)

### Operations Research & Reinforcement Learning
  - Scaling Average-reward Reinforcement Learning for Product Delivery (Proper, AAAI 2004) [[Paper]](http://web.engr.oregonstate.edu/~proper/AAAI04SProper.pdf)
  - Cross Channel Optimized Marketing by Reinforcement Learning (Abe, KDD 2004) [[Paper]](http://www.research.ibm.com/people/n/nabe/kdd04AVAS.pdf)

### Human Computer Interaction
  - Optimizing Dialogue Management with Reinforcement Learning: Experiments with the NJFun System (Singh, JAIR 2002) [[Paper]](http://web.eecs.umich.edu/~baveja/Papers/RLDSjair.pdf)

## Reinforcement Learning Tutorials / Websites

  - Mance Harmon and Stephanie Harmon, [Reinforcement Learning: A Tutorial](http://old.nbu.bg/cogs/events/2000/Readings/Petrov/rltutorial.pdf)
  - C. Igel, M.A. Riedmiller, et al., Reinforcement Learning in a Nutshell, ESANN, 2007. [[Paper]](http://image.diku.dk/igel/paper/RLiaN.pdf)
  - UNSW - [Reinforcement Learning](http://www.cse.unsw.edu.au/~cs9417ml/RL1/index.html)
   - [Introduction](http://www.cse.unsw.edu.au/~cs9417ml/RL1/introduction.html)
   - [TD-Learning](http://www.cse.unsw.edu.au/~cs9417ml/RL1/tdlearning.html)
   - [Q-Learning and SARSA](http://www.cse.unsw.edu.au/~cs9417ml/RL1/algorithms.html)
   - [Applet for "Cat and Mouse" Game](http://www.cse.unsw.edu.au/~cs9417ml/RL1/applet.html)
  - [ROS Reinforcement Learning Tutorial](http://wiki.ros.org/reinforcement_learning/Tutorials/Reinforcement%20Learning%20Tutorial)
  - [POMDP for Dummies](http://cs.brown.edu/research/ai/pomdp/tutorial/index.html)
  - Scholarpedia articles on:
   - [Reinforcement Learning](http://www.scholarpedia.org/article/Reinforcement_learning)
   - [Temporal Difference Learning](http://www.scholarpedia.org/article/Temporal_difference_learning)
  - Repository with useful [MATLAB Software, presentations, and demo videos](http://busoniu.net/repository.php)
  - [Bibliography on Reinforcement Learning](http://liinwww.ira.uka.de/bibliography/Neural/reinforcement.learning.html)
  - UC Berkeley - CS 294: Deep Reinforcement Learning, Fall 2015 (John Schulman, Pieter Abbeel) [[Class Website]](http://rll.berkeley.edu/deeprlcourse/)
  - [Blog posts on Reinforcement Learning, Parts 1-4](https://studywolf.wordpress.com/2012/11/25/reinforcement-learning-q-learning-and-exploration/) by Travis DeWolf
  - [The Arcade Learning Environment](http://www.arcadelearningenvironment.org/) - Atari 2600 games environment for developing AI agents
  - [Deep Reinforcement Learning: Pong from Pixels](http://karpathy.github.io/2016/05/31/rl/) by Andrej Karpathy
  - [Demystifying Deep Reinforcement Learning](https://www.nervanasys.com/demystifying-deep-reinforcement-learning/)
  - [Let’s make a DQN](https://jaromiru.com/2016/09/27/lets-make-a-dqn-theory/)
  - [Simple Reinforcement Learning with Tensorflow, Parts 0-8](https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-0-q-learning-with-tables-and-neural-networks-d195264329d0#.78km20i8r) by Arthur Juliani
  - [Practical_RL](https://github.com/yandexdataschool/Practical_RL) - github-based course in reinforcement learning in the wild (lectures, coding labs, projects)


## Online Demos
 - [Deep Q-Learning Demo](http://cs.stanford.edu/people/karpathy/convnetjs/demo/rldemo.html) - A deep Q learning demonstration using ConvNetJS
 - [Deep Q-Learning with Tensor Flow](https://github.com/nivwusquorum/tensorflow-deepq) - A deep Q learning demonstration using Google Tensorflow
 - [Reinforcement Learning Demo](http://cs.stanford.edu/people/karpathy/reinforcejs/) - A reinforcement learning demo using reinforcejs by Andrej Karpathy
