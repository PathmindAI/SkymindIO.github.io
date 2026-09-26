---
title: 생성적 적대 신경망(GANs)에 대한 초보자용 가이드 (GANs)
author: Chris V. Nicholson
short_title: Generative Adversarial Network (GAN)
description: 생성적 적대 신경망(GANs)이란 한 네트워크가 다른 네트워크와 겨루는 두개의 네트워크로 구성된 심층 신경망 구조이다.
---

*프로그래머가 예술가가 아니라고 생각하겠지만, 프로그래밍은 굉장히 창의적인 영역이다. 프로그래밍은 논리 기반의 창작물이다. - John Romero*

생성적 적대 신경망(GANs)이란 두개의 네트워크로 구성된 심층 신경망 구조이다. 한 네트워크가 다른 네트워크와 겨루는 구조이기 때문에 적대적(Adversarial)이라는 의미가 이름에 포함되었다.


GANs는 2014년 몬트리올 대학의 Ian Goodfellow와 Yoshua Bengio를 포함한 다른 연구자들이 작성한 [논문에서 소개되었다](https://arxiv.org/abs/1406.2661){:target="_blank"}. 
Facebook의 인공지능 연구 책임자인 Yann LeCun은  GANs의 [adversarial training](https://www.quora.com/What-are-some-recent-and-potentially-upcoming-breakthroughs-in-deep-learning){:target="_blank"}을 “머신 러닝 분야에서 지난 10년 간 가장 흥미로운 아이디어”라고 언급했다.


어떠한 분포의 데이터도 모방하는 학습을 할 수 있기 때문에 GANs의 잠재능력은 엄청나다고 할 수 있다. 
즉 GANs은 이미지, 음악, 연설, 산문 등의 모든 분야에서 실제 우리의 세계와 놀랍도록 비슷한 새로운 세계를 창조하도록 학습될 수 있다. 
그들은 어떤 의미에서는 로봇 예술가이며 그 결과물은 [매우 인상적이다.](https://www.nytimes.com/2017/08/14/arts/design/google-how-ai-creates-new-music-and-new-artists-project-magenta.html){:target="_blank"}

[크리스티 경매에서](https://www.theverge.com/2018/10/23/18013190/ai-art-portrait-auction-christies-belamy-obvious-robbie-barrat-gans) [스탠포드의 Robbie Barrat이 작성한 오픈 소스 코드](https://github.com/robbiebarrat/art-DCGAN)를 활용한 GAN이 만든 초상화가 432,000달러에 낙찰되었다. 이 초상화는 프랑스의 3명의 개발자그룹인 Obvious가 탄생시킨 작품으로, 크리스티 경매에서 최초로 판매된 AI가 만든 그림이다.

![AI painter](/images/wiki/aipainter.jpg)

## Generative vs. Discriminative Algorithms

GANs을 이해하려면 생성(generative) 알고리즘이 작동하는 방식을 알아야 한다. 이는 판별(discriminative) 알고리즘과 대조하여 더욱 쉽게 이해할 수 있다. 판별 알고리즘은 주로 입력 데이터를 분류하는 것에 집중한다. 즉 데이터 인스턴스의 피쳐가 주어지면 해당 데이터가 속한 레이블 또는 카테고리를 예측하게 된다.

예를 들면, 판별 알고리즘은 이메일의 모든 단어를 판별하여 메시지가 `스팸(spam)`인지 `스팸이 아닌지(not_spam)`를 예측할 수 있다. 여기서 `스팸(spam)`은 하나의 레이블이고 이메일에서 수집 된 BoW(Bag of Words)는 입력 데이터를 구성하는 피쳐이다. 이 문제를 수학적으로 표현하면 레이블은 `y`가 되고 피쳐는 `x`가 된다. 공식 `p(y|x)`는 "주어진 x에 대한 y의 확률"을 의미하고, 이 경우 “이메일에 포함된 단어들이 주어졌을때 그 이메일이 스팸일 확률”로 해석된다.

따라서 판별 알고리즘은 피쳐를 레이블에 매핑한다. 즉, 피쳐와 레이블간의 상관관계에만 집중하게 된다. 반면 생성 알고리즘은 이와 반대로 작동한다. 생성 알고리즘은 주어진 특정 피쳐로부터 레이블을 예측하는 대신 주어진 특정 레이블에서 피쳐를 예측하는 방식이다.

생성 알고리즘이 대답하려는 질문은 다음과 같다. 이메일이 스팸이라고 가정했을 때, 그럴 가능성이 있는 피쳐가 얼마나 있을까? 판별 모델이 `y`와 `x`의 관계를 중요시한다면, 생성 모델은 “x를 얻는 방법”을 중요시한다. 이를 통해 주어진 `y`에 대한 `x`의 확률인 `p(x|y)` 또는 주어진 레이블 또는 카테고리에 대한 피쳐일 확률을 알 수 있게 해준다. 다시 말해 생성 알고리즘은 입력 데이터를 단순히 분류하는 것 이상의 기능을 수행할 수 있다. 

생성 알고리즘과 판별 알고리즘을 구분하는 또 다른 방법은 다음과 같다.

* 판별 알고리즘은 클래스 간의 경계를 학습한다.
* 생성 알고리즘은 각 클래스의 분포를 모델링한다.

<a class="w-button button-skilcta" href="https://nbviewer.jupyter.org/github/SkymindIO/skil-python/blob/master/examples/keras-skil-example.ipynb" style="width:75%; margin-top: 15px;" target="_blank">Get Started with AI</a>

## GAN의 작동 원리

앞서 언급한 바와 같이 GANs은 두 개의 신경망으로 구성되어 있는데, *generator*라고 불리는 하나의 신경망이 새로운 데이터 인스턴스를 생성하고 다른 하나인 *discriminator*는 데이터의 진위를 평가한다. 즉 discriminator는 검토한 각각의 데이터 인스턴스가 실제 트레이닝 데이터 세트인지 아닌지를 판단하게 된다.

예를 들어, MNIST 데이터 세트와 같은 손글씨 숫자를 생성한다고 가정 했을 때 discriminator의 목표는 실제 MNIST 데이터세트 인스턴스를 보여줄 때 그것이 진짜인지 알아내는 것이다.  

한편 generator는 discriminator에게 전달하는 새로운 이미지를 생성한다. generator의 입장에서는 생성한 가짜 이미지가 진짜 이미지인것처럼 보여지길 원한다. 여기에서 generator의 목표는 discriminator가 진짜로 판별하게 만드는 손글씨 숫자를 생성하는 것이다. 반면 discriminator의 목표는 generator로 부터 전달된 이미지를 가짜로 식별하는 것이다.

GANs의 동작 단계는 다음과 같다.

* generator가 임의의 수를 입력받아 생성한 이미지로 반환한다.
* 이렇게 생성된 이미지는 실제 데이터 세트에서 가져온 이미지들과 함께 discriminator에 전달된다.
* discriminator는 실제 이미지와 가짜 이미지를 판별하여 0과 1 사이의 확률값으로 반환한다. 1은 실제 이미지, 0은 가짜 이미지를 의미한다.

따라서 이중 피드백 루프를 갖게 된다.

* discriminator는 이미지의 정답 값으로부터 피드백을 받는다.
* generator는 discriminator로 부터 피드백을 받는다.


![GAN schema](/images/wiki/gan_schema.png)
Credit: O'Reilly

Ian Goodfellow의 “Generative Adversarial Network(2014)” 논문에서 GANs을 지폐위조범과 경찰에 비유했다. 지폐위조범(generator)은 더욱 교묘하게 속이려고 하고 경찰(discriminator)은 이렇게 위조된 지폐를 감별(classify)하려고 한다. 때문에 양쪽 모두 점진적으로 변화하여 결국 두 그룹 모두 속이고 감별하는 서로의 능력이 발전하게 된다.

discriminator 네트워크가 전달된 이미지를 실제 이미지인지 가짜 이미지인지를 판별할 수 있는 일반적인 컨볼루션 네트워크라면 generator는 정반대의 컨볼루션 네트워크이다. 일반 컨볼루션 분류기는 이미지를 입력받아 확률을 예측하기 위해 이를 다운 샘플링하는 반면 generator는 랜덤 노이즈 벡터를 입력받아 이미지를 만드는 업샘플링을 한다. 즉 일반적인 컨볼루션 네트워크는 maxpooling과 같은 다운 샘플링 기술을 사용하여 데이터를 처리하고 generator와 같은 inverse 컨볼루션 네트워크는 새로운 데이터를 생성한다.

두 네트워크 모두 제로섬 게임처럼 서로 반대되는 목적함수 또는 손실함수를 통해 최적화하려고 시도한다. 이것은 [actor-critic model](https://arxiv.org/abs/1610.01945){:target="_blank"}과 비슷하다.
discriminator의 행동이 바뀌면 generator의 행동도 변하고 그 반대의 경우도 마찬가지이다. 각 에러는 서로에게 전달된다.

![GANs](/images/wiki/GANs.png)
*Image credit: [Thalles Silva](https://medium.freecodecamp.org/an-intuitive-introduction-to-generative-adversarial-networks-gans-7a2264a81394)*

이미지 생성에 대해 더 알고 싶다면 Brandon Amos가 작성한 [이미지를 확률 분포의 샘플로 해석하는 것에 대한 포스팅](http://bamos.github.io/2016/08/09/deep-completion/#step-1-interpreting-images-as-samples-from-a-probability-distribution){:target="_blank"}을 참고하길 바란다.

## GANs, Autoencoders and VAEs
생성적 적대 신경망(generative adversarial networks)을 autoencoders와 VAEs와 같은 신경망에 비교해보자.

Autoencoders는 입력 데이터를 벡터로 암호화(encode) 한다. 미가공(raw) 데이터를 잠재된 차원(hidden representation)으로 압축한다는 것이다. 즉, 잠재된 차원의 벡터는 미가공 데이터를 훨씬 낮은 차원으로 축소한다. Autoencoders는 decoder와 함께 사용되는데, decoder는 Boltzmann 분포를 활용한 [Restricted Boltzmann Machines(RBM)](./restricted-boltzmann-machine)처럼 잠재된 차원의 입력 데이터를 재구성(reconstruct)한다.

![autencoder schema](/images/wiki/autoencoder_schema.jpg)
[Credit: Keras blog](https://blog.keras.io/building-autoencoders-in-keras.html){:target="_blank"}

Variational autoencoder는 입력 데이터 인코딩에 제약 조건을 추가하는 생성 알고리즘이다. 즉 잠재된 차원이 정규화 된다. Variational autoencoder는 데이터를 압축하는 autoencoder와 데이터를 생성하는 GAN, 두 가지 역할을 수행할 수 있다. 그러나 세부적인 데이터를 생성하는 GAN에 비해 VAE가 생성한 이미지는 더 흐린 경향이 있다. 상세한 구현은 Deeplearning4j의 [autoencoders와 variational autoencoders](https://github.com/deeplearning4j/dl4j-examples/tree/master/dl4j-examples/src/main/java/org/deeplearning4j/examples/unsupervised){:target="_blank"} 예제를 참고하기 바란다.

생성 알고리즘은 세 가지 유형으로 나뉜다.

* 레이블이 주어지면 관련 피쳐를 예측한다.(Naive Bayes)
* 잠재된 차원(hidden representation)으로 관련 피쳐를 예측한다.(VAE, GAN)
* 주어진 피쳐들로 나머지(inpainting, imputation)를 예측한다.

## GAN 트레이닝  팁

discriminator를 트레이닝할 때 generator의 파라미터 값을 고정 시키고, generator를 트레이닝할 때는 discriminator의 파라미터 값을 고정시킨다. discriminator와 generator는 각각 고정된 상대 네트워크의 결과값을 통해 학습하게 된다. 예를 들어, 이로 인해 generator가 반드시 학습해야 하는 gradient를 더 잘 반영할 수 있게 된다.

다음으로, generator 트레이닝을 시작하기 전에 discriminator를 MNIST 데이터를 통해 미리 학습시키면 더 명확한 gradient값을 얻을 수 있다.

GAN의 두 개의 네트워크를 트레이닝 하다 보면 다음과 같은 문제가 생길 수 있다. discriminator가 너무 뛰어나면 0이나 1에 매우 가까운 gradient값을 반환하게 되어, generator가 gradient값을 제대로 반영하기 어렵게 된다. generator가 너무 뛰어나면 discriminator가 진짜 데이터를 가짜 데이터로 판단할 확률이 높아진다. 이러한 문제는 두 신경망의 학습률(learning rates)을 각각 설정하여 완화할 수 있다. 두 개의 신경망은 항상 비슷한 "학습 수준" <sup>[1](#one)</sup>을 유지해야 한다.

일반적으로 GAN을 트레이닝하는데는 오랜 시간이 걸린다. 하나의 GPU에서는 몇 시간이 걸릴 수도, CPU에서는 하루 이상이 걸릴 수도 있다. 이러한 GAN의 학습과 사용의 어려움에도 많은 [흥미로운 연구와 글](https://blog.openai.com/generative-models/){:target="_blank"}이 계속되고 있다.


## 코드 예제 

다음은 [Keras의 GAN의 예제](https://github.com/eriklindernoren/Keras-GAN/blob/master/gan/gan.py){:target="_blank"}이다. 이 모델은 Deeplearning4j 모델 가져오기(model importing)기능을 통해 Deeplearning4j에서 사용할 수 있다.

```
class GAN():
    def __init__(self):
        self.img_rows = 28
        self.img_cols = 28
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)

        optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy'])

        # Build and compile the generator
        self.generator = self.build_generator()
        self.generator.compile(loss='binary_crossentropy', optimizer=optimizer)

        # The generator takes noise as input and generated imgs
        z = Input(shape=(100,))
        img = self.generator(z)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The valid takes generated images as input and determines validity
        valid = self.discriminator(img)

        # The combined model  (stacked generator and discriminator) takes
        # noise as input => generates images => determines validity
        self.combined = Model(z, valid)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)

    def build_generator(self):

        noise_shape = (100,)

        model = Sequential()

        model.add(Dense(256, input_shape=noise_shape))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(1024))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(np.prod(self.img_shape), activation='tanh'))
        model.add(Reshape(self.img_shape))

        model.summary()

        noise = Input(shape=noise_shape)
        img = model(noise)

        return Model(noise, img)

    def build_discriminator(self):

        img_shape = (self.img_rows, self.img_cols, self.channels)

        model = Sequential()

        model.add(Flatten(input_shape=img_shape))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(256))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(1, activation='sigmoid'))
        model.summary()

        img = Input(shape=img_shape)
        validity = model(img)

        return Model(img, validity)

    def train(self, epochs, batch_size=128, save_interval=50):

        # Load the dataset
        (X_train, _), (_, _) = mnist.load_data()

        # Rescale -1 to 1
        X_train = (X_train.astype(np.float32) - 127.5) / 127.5
        X_train = np.expand_dims(X_train, axis=3)

        half_batch = int(batch_size / 2)

        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random half batch of images
            idx = np.random.randint(0, X_train.shape[0], half_batch)
            imgs = X_train[idx]

            noise = np.random.normal(0, 1, (half_batch, 100))

            # Generate a half batch of new images
            gen_imgs = self.generator.predict(noise)

            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch(imgs, np.ones((half_batch, 1)))
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, np.zeros((half_batch, 1)))
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)


            # ---------------------
            #  Train Generator
            # ---------------------

            noise = np.random.normal(0, 1, (batch_size, 100))

            # The generator wants the discriminator to label the generated samples
            # as valid (ones)
            valid_y = np.array([1] * batch_size)

            # Train the generator
            g_loss = self.combined.train_on_batch(noise, valid_y)

            # Plot the progress
            print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))

            # If at save interval => save generated image samples
            if epoch % save_interval == 0:
                self.save_imgs(epoch)

    def save_imgs(self, epoch):
        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, 100))
        gen_imgs = self.generator.predict(noise)

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap='gray')
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig("gan/images/mnist_%d.png" % epoch)
        plt.close()


if __name__ == '__main__':
    gan = GAN()
    gan.train(epochs=30000, batch_size=32, save_interval=200)
```

![GAN dancers](/images/wiki/GANdancers.png)

*Credit: The New Yorker*

## <a name="footnote">Footnotes</a>

<a name="one">1)</a> *이런 관점에서 진화에 대해 생각해보면, 한편으로는 유전적 돌연변이를, 또 다른 한편으로는 자연선택이 두 개의 대립적 알고리즘으로 작용한다는 것이다. 인류세(Anthropocene)에서 우리가 목격하고 있는 것은 진화 알고리즘의 절반은 다른 종에 비해 승리했다는 것이다. 한가지 종(species)의 유전적 돌연변이인 호모사피엔스는 강력한 도구를 만들어 사용하였고 인간을 형성하는데 미치는 자연 선택의 역할은 매우 작아지게 되었다. 일부 사람들은 그러한 불균형이 잘못 트레이닝된 GAN에서 볼 수 있는 현상처럼 시스템의 심각한 붕괴로 이어지고 있다고 생각할 것이다. 한 걸음 더 나아가자면, 이것은 아마도 대여과기(a Great Filter)와 유사한 지적 생명체의 발달에 있어서의 구조적 결함일 것인데, 이는 왜 인간이 그토록 큰 우주에서 진화된 생명체가 생겨야 할 수학적인 확률에도 불구하고 다른 선진 종의 흔적을 발견하지 못했는지를 설명해 준다.*

*호모 사피엔스는 자원을 얻기 위해 경쟁하는 다른 종보다 빠르게 진화하고 있다. 이런 식으로 보면, 우리는 주로 속도에 관한 지능의 정의에 대해 고려해볼 수 있다. 다른 모든 것들이 같다는 조건 하에, 더 지능적인 유기체(또는 종이나 알고리즘)가 더 적은 시간 안에 같은 문제를 해결한다. 현재 AI 분야에서 볼 수 있는 것은 알고리즘이 보다 빠른 칩, 병렬 계산, 수억 달러의 연구 자금에 힘입어 점점 더 많은 문제를 해결하는 능력을 가속화하는 것이다. Elon Musk는 AI에 대한 우려를 표명했지만 명확한 근거를 바탕으로 충분하게 표현하지는 않았다. 인간이 멸종으로 치닫고 있는 종보다 더 빨리 배우듯이 알고리즘이 인간보다 더 빨리 학습하고 있다. 속도와 관련이 있다는 말인데, 대규모로 병렬화된 하드웨어는 시간을 병렬화하는 방법이다. 그리고 그것은 인간의 뇌가 아직 해내지 못한 것이다.*

## 생성 네트워크용 리소스

* ["Generative Learning algorithms" - Andrew Ng's Stanford notes](http://cs229.stanford.edu/notes/cs229-notes2.pdf)
* [On Discriminative vs. Generative classifiers: A comparison of logistic regression and naive Bayes, by Andrew Ng and Michael I. Jordan](http://papers.nips.cc/paper/2020-on-discriminative-vs-generative-classifiers-a-comparison-of-logistic-regression-and-naive-bayes.pdf)
* [The Math Behind Generative Adversarial Networks](https://lilianweng.github.io/lil-log/2017/08/20/from-GAN-to-WGAN.html)

![stained glass cycle GAN](/images/wiki/stained_glass_cycleGAN.jpg)

## GAN 사용 사례

* [텍스트를 이미지로 생성](https://arxiv.org/abs/1605.05396)
* [이미지에서 이미지로 변환](https://arxiv.org/abs/1611.07004)
* [이미지 해상도 증가](https://arxiv.org/abs/1609.04802)
* [다음 비디오 프레임 예측]( https://arxiv.org/abs/1511.06380)

## GANs의 주요 논문 

* [Generative Adversarial Nets] [[Paper]](https://arxiv.org/abs/1406.2661)
[[Code]](https://github.com/goodfeli/adversarial)(Ian Goodfellow's breakthrough paper)

### 그 밖의 논문과 자료

* [GAN Hacks: How to Train a GAN? Tips and tricks to make GANs work](https://github.com/soumith/ganhacks)

* Deep Generative Image Models using a Laplacian Pyramid of Adversarial Networks] [[Paper]](https://arxiv.org/abs/1506.05751)[[Code]](https://github.com/facebook/eyescream)

* [Adversarial Autoencoders] [[Paper]](http://arxiv.org/abs/1511.05644)[[Code]](https://github.com/musyoku/adversarial-autoencoder)

* [Generating Images with Perceptual Similarity Metrics based on Deep Networks] [[Paper]](https://arxiv.org/pdf/1602.02644v2.pdf)

* [Generating images with recurrent adversarial networks] [[Paper]](https://arxiv.org/abs/1602.05110)[[Code]](https://github.com/ofirnachum/sequence_gan)

* [Generative Visual Manipulation on the Natural Image Manifold] [[Paper]](https://people.eecs.berkeley.edu/~junyanz/projects/gvm/eccv16_gvm.pdf)[[Code]](https://github.com/junyanz/iGAN)

* [Learning What and Where to Draw] [[Paper]](http://www.scottreed.info/files/nips2016.pdf)[[Code]](https://github.com/reedscot/nips2016)

* [Adversarial Training for Sketch Retrieval] [[Paper]](http://link.springer.com/chapter/10.1007/978-3-319-46604-0_55)

* [Generative Image Modeling using Style and Structure Adversarial Networks] [[Paper]](https://arxiv.org/pdf/1603.05631.pdf)[[Code]](https://github.com/xiaolonw/ss-gan)

* [Generative Adversarial Networks as Variational Training of Energy Based Models] [[Paper]](http://www.mathpubs.com/detail/1611.01799v1/Generative-Adversarial-Networks-as-Variational-Training-of-Energy-Based-Models)(ICLR 2017)

* [Synthesizing the preferred inputs for neurons in neural networks via deep generator networks] [[Paper]](https://arxiv.org/pdf/1605.09304v5.pdf)[[Code]](https://github.com/Evolving-AI-Lab/synthesizing)

* [SalGAN: Visual Saliency Prediction with Generative Adversarial Networks] [[Paper]](https://arxiv.org/abs/1701.01081)[[Code]](https://github.com/imatge-upc/saliency-salgan-2017)

* [Adversarial Feature Learning] [[Paper]](https://arxiv.org/abs/1605.09782)

### 고화질 이미지 생성

* [Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks] [[Paper]](https://arxiv.org/abs/1511.06434)[[Code]](https://github.com/jacobgil/keras-dcgan)(Gan with convolutional networks)(ICLR)

* [Generative Adversarial Text to Image Synthesis] [[Paper]](https://arxiv.org/abs/1605.05396)[[Code]](https://github.com/reedscot/icml2016)[[Code]](https://github.com/paarthneekhara/text-to-image)

* [Improved Techniques for Training GANs] [[Paper]](https://arxiv.org/abs/1606.03498)[[Code]](https://github.com/openai/improved-gan)(Goodfellow's paper)

* [Plug & Play Generative Networks: Conditional Iterative Generation of Images in Latent Space] [[Paper]](https://arxiv.org/abs/1612.00005v1)[[Code]](https://github.com/Evolving-AI-Lab/ppgn)

* [StackGAN: Text to Photo-realistic Image Synthesis with Stacked Generative Adversarial Networks] [[Paper]](https://arxiv.org/pdf/1612.03242v1.pdf)[[Code]](https://github.com/hanzhanggit/StackGAN)

* [Improved Training of Wasserstein GANs] [[Paper]](https://arxiv.org/abs/1704.00028)[[Code]](https://github.com/igul222/improved_wgan_training)

* [Boundary Equibilibrium Generative Adversarial Networks Implementation in Tensorflow] [[Paper]](https://arxiv.org/abs/1703.10717)[[Code]](https://github.com/artcg/BEGAN)

* [Progressive Growing of GANs for Improved Quality, Stability, and Variation ] [[Paper]](http://research.nvidia.com/publication/2017-10_Progressive-Growing-of)[[Code]](https://github.com/tkarras/progressive_growing_of_gans)

### 준 지도 학습(Semi-supervised learning)

* [Adversarial Training Methods for Semi-Supervised Text Classification] [[Paper]](https://arxiv.org/abs/1605.07725)[[Note]](https://github.com/dennybritz/deeplearning-papernotes/blob/master/notes/adversarial-text-classification.md)( Ian Goodfellow Paper)

* [Improved Techniques for Training GANs] [[Paper]](https://arxiv.org/abs/1606.03498)[[Code]](https://github.com/openai/improved-gan)(Goodfellow's paper)

* [Unsupervised and Semi-supervised Learning with Categorical Generative Adversarial Networks] [[Paper]](https://arxiv.org/abs/1511.06390)(ICLR)

* [Semi-Supervised QA with Generative Domain-Adaptive Nets] [[Paper]](https://arxiv.org/abs/1702.02206)(ACL 2017)

### 앙상블(Ensembles)

* [AdaGAN: Boosting Generative Models] [[Paper]](https://arxiv.org/abs/1701.02386)[[Code]]（Google Brain）

### 클러스터링(Clustering)

* [Unsupervised and Semi-supervised Learning with Categorical Generative Adversarial Networks] [[Paper]](https://arxiv.org/abs/1511.06390)(ICLR)

### Image blending

* [GP-GAN: Towards Realistic High-Resolution Image Blending] [[Paper]](https://arxiv.org/abs/1703.07195)[[Code]](https://github.com/wuhuikai/GP-GAN)

### Image Inpainting

* [Semantic Image Inpainting with Perceptual and Contextual Losses] [[Paper]](https://arxiv.org/abs/1607.07539)[[Code]](https://github.com/bamos/dcgan-completion.tensorflow)(CVPR 2017)

* [Context Encoders: Feature Learning by Inpainting] [[Paper]](https://arxiv.org/abs/1604.07379)[[Code]](https://github.com/jazzsaxmafia/Inpainting)

* [Semi-Supervised Learning with Context-Conditional Generative Adversarial Networks] [[Paper]](https://arxiv.org/abs/1611.06430v1)

* [Generative face completion] [[Paper]](https://drive.google.com/file/d/0B8_MZ8a8aoSeenVrYkpCdnFRVms/edit)[[Code]](https://github.com/Yijunmaverick/GenerativeFaceCompletion)(CVPR2017)

* [Globally and Locally Consistent Image Completion] [[MainPAGE]](http://hi.cs.waseda.ac.jp/~iizuka/projects/completion/en/)(SIGGRAPH 2017)

### 결합확률(Joint Probability)

* [Adversarially Learned Inference][[Paper]](https://arxiv.org/abs/1606.00704)[[Code]](https://github.com/IshmaelBelghazi/ALI)

### Super-Resolution

* [Image super-resolution through deep learning ][[Code]](https://github.com/david-gpu/srez)(Just for face dataset)

* [Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network] [[Paper]](https://arxiv.org/abs/1609.04802)[[Code]](https://github.com/leehomyc/Photo-Realistic-Super-Resoluton)（Using Deep residual network）

* [EnhanceGAN] [Docs](https://medium.com/@richardherbert/faces-from-noise-super-enhancing-8x8-images-with-enhancegan-ebda015bb5e0#.io6pskvin)[[Code]]

### De-occlusion

* [Robust LSTM-Autoencoders for Face De-Occlusion in the Wild] [[Paper]](https://arxiv.org/abs/1612.08534)

### Semantic Segmentation

* [Adversarial Deep Structural Networks for Mammographic Mass Segmentation] [[Paper]](https://arxiv.org/abs/1612.05970)[[Code]](https://github.com/wentaozhu/adversarial-deep-structural-networks)

* [Semantic Segmentation using Adversarial Networks] [[Paper]](https://arxiv.org/abs/1611.08408)（Soumith's paper）

### Object Detection

* [Perceptual generative adversarial networks for small object detection] [[Paper]](https://arxiv.org/abs/1706.05274v2)(CVPR 2017)

* [A-Fast-RCNN: Hard Positive Generation via Adversary for Object Detection] [[Paper]](http://abhinavsh.info/papers/pdfs/adversarial_object_detection.pdf)[[Code]](https://github.com/xiaolonw/adversarial-frcnn)(CVPR2017)

### RNN-GANs

* [C-RNN-GAN: Continuous recurrent neural networks with adversarial training] [[Paper]](https://arxiv.org/abs/1611.09904)[[Code]](https://github.com/olofmogren/c-rnn-gan)

### Conditional Adversarial Nets

* [Conditional Generative Adversarial Nets] [[Paper]](https://arxiv.org/abs/1411.1784)[[Code]](https://github.com/zhangqianhui/Conditional-Gans)

* [InfoGAN: Interpretable Representation Learning by Information Maximizing Generative Adversarial Nets] [[Paper]](https://arxiv.org/abs/1606.03657)[[Code]](https://github.com/buriburisuri/supervised_infogan)[[Code]](https://github.com/openai/InfoGAN)

* [Conditional Image Synthesis With Auxiliary Classifier GANs] [[Paper]](https://arxiv.org/abs/1610.09585)[[Code]](https://github.com/buriburisuri/ac-gan)(GoogleBrain ICLR 2017)

* [Pixel-Level Domain Transfer] [[Paper]](https://arxiv.org/pdf/1603.07442v2.pdf)[[Code]](https://github.com/fxia22/pldtgan)

* [Invertible Conditional GANs for image editing] [[Paper]](https://arxiv.org/abs/1611.06355)[[Code]](https://github.com/Guim3/IcGAN)

* [Plug & Play Generative Networks: Conditional Iterative Generation of Images in Latent Space] [[Paper]](https://arxiv.org/abs/1612.00005v1)[[Code]](https://github.com/Evolving-AI-Lab/ppgn)

* [StackGAN: Text to Photo-realistic Image Synthesis with Stacked Generative Adversarial Networks] [[Paper]](https://arxiv.org/pdf/1612.03242v1.pdf)[[Code]](https://github.com/hanzhanggit/StackGAN)

* [MaskGAN: Better Text Generation via Filling in the ______](https://arxiv.org/abs/1801.07736) Goodfellow et al


### 비디오 예측 및 생성

* [Deep multi-scale video prediction beyond mean square error] [[Paper]](https://arxiv.org/abs/1511.05440)[[Code]](https://github.com/dyelax/Adversarial_Video_Generation)(Yann LeCun's paper)

* [Generating Videos with Scene Dynamics] [[Paper]](https://arxiv.org/abs/1609.02612)[[Web]](http://web.mit.edu/vondrick/tinyvideo/)[[Code]](https://github.com/cvondrick/videogan)

* [MoCoGAN: Decomposing Motion and Content for Video Generation] [[Paper]](https://arxiv.org/abs/1707.04993)

### Texture Synthesis & Style Transfer

* [Precomputed real-time texture synthesis with markovian generative adversarial networks] [[Paper]](https://arxiv.org/abs/1604.04382)[[Code]](https://github.com/chuanli11/MGANs)(ECCV 2016)

### Image Translation

* [Unsupervised cross-domain image generation] [[Paper]](https://arxiv.org/abs/1611.02200)[[Code]](https://github.com/yunjey/domain-transfer-network)

* [Image-to-image translation using conditional adversarial nets] [[Paper]](https://arxiv.org/pdf/1611.07004v1.pdf)[[Code]](https://github.com/phillipi/pix2pix)[[Code]](https://github.com/yenchenlin/pix2pix-tensorflow)

* [Learning to Discover Cross-Domain Relations with Generative Adversarial Networks] [[Paper]](https://arxiv.org/abs/1703.05192)[[Code]](https://github.com/carpedm20/DiscoGAN-pytorch)

* [Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks] [[Paper]](https://junyanz.github.io/CycleGAN/)[[Code]](https://github.com/junyanz/CycleGAN)

* [CoGAN: Coupled Generative Adversarial Networks] [[Paper]](https://arxiv.org/abs/1606.07536)[[Code]](https://github.com/andrewliao11/CoGAN-tensorflow)(NIPS 2016)

* [Unsupervised Image-to-Image Translation with Generative Adversarial Networks] [[Paper]](https://arxiv.org/pdf/1701.02676.pdf)

* [Unsupervised Image-to-Image Translation Networks] [[Paper]](https://arxiv.org/abs/1703.00848)

* [Triangle Generative Adversarial Networks] [[Paper]](https://arxiv.org/abs/1709.06548)


### GAN 이론 

* [Energy-based generative adversarial network] [[Paper]](https://arxiv.org/pdf/1609.03126v2.pdf)[[Code]](https://github.com/buriburisuri/ebgan)(Lecun paper)

* [Improved Techniques for Training GANs] [[Paper]](https://arxiv.org/abs/1606.03498)[[Code]](https://github.com/openai/improved-gan)(Goodfellow's paper)

* [Mode Regularized Generative Adversarial Networks] [[Paper]](https://openreview.net/pdf?id=HJKkY35le)(Yoshua Bengio , ICLR 2017)

* [Improving Generative Adversarial Networks with Denoising Feature Matching] [[Paper]](https://openreview.net/pdf?id=S1X7nhsxl)[[Code]](https://github.com/hvy/chainer-gan-denoising-feature-matching)(Yoshua Bengio , ICLR 2017)

* [Sampling Generative Networks] [[Paper]](https://arxiv.org/abs/1609.04468)[[Code]](https://github.com/dribnet/plat)

* [How to train Gans] [[Docu]](https://github.com/soumith/ganhacks#authors)

* [Towards Principled Methods for Training Generative Adversarial Networks] [[Paper]](http://openreview.net/forum?id=Hk4_qw5xe)(ICLR 2017)

* [Unrolled Generative Adversarial Networks] [[Paper]](https://arxiv.org/abs/1611.02163)[[Code]](https://github.com/poolio/unrolled_gan)(ICLR 2017)

* [Least Squares Generative Adversarial Networks] [[Paper]](https://arxiv.org/abs/1611.04076)[[Code]](https://github.com/pfnet-research/chainer-LSGAN)(ICCV 2017)

* [Wasserstein GAN] [[Paper]](https://arxiv.org/abs/1701.07875)[[Code]](https://github.com/martinarjovsky/WassersteinGAN)

* [Improved Training of Wasserstein GANs] [[Paper]](https://arxiv.org/abs/1704.00028)[[Code]](https://github.com/igul222/improved_wgan_training)(The improve of wgan)

* [Towards Principled Methods for Training Generative Adversarial Networks] [[Paper]](https://arxiv.org/abs/1701.04862)

* [Generalization and Equilibrium in Generative Adversarial Nets] [[Paper]](https://arxiv.org/abs/1703.00573)（ICML 2017）


### 3D GANs

* [Learning a Probabilistic Latent Space of Object Shapes via 3D Generative-Adversarial Modeling] [[Paper]](https://arxiv.org/abs/1610.07584)[[Web]](http://3dgan.csail.mit.edu/)[[Code]](https://github.com/zck119/3dgan-release)(2016 NIPS)

* [Transformation-Grounded Image Generation Network for Novel 3D View Synthesis] [[Web]](http://www.cs.unc.edu/%7Eeunbyung/tvsn/)(CVPR 2017)

### 음악 

* [MidiNet: A Convolutional Generative Adversarial Network for Symbolic-domain Music Generation using 1D and 2D Conditions] [[Paper]](https://arxiv.org/abs/1703.10847)[[HOMEPAGE]](https://richardyang40148.github.io/TheBlog/midinet_arxiv_demo.html)

### 얼굴 생성 및 편집

* [Autoencoding beyond pixels using a learned similarity metric] [[Paper]](https://arxiv.org/abs/1512.09300)[[Code]](https://github.com/andersbll/autoencoding_beyond_pixels)[[Tensorflow code]](https://github.com/zhangqianhui/vae-gan-tensorflow)

* [Coupled Generative Adversarial Networks] [[Paper]](http://mingyuliu.net/)[[Caffe Code]](https://github.com/mingyuliutw/CoGAN)[[Tensorflow Code]](https://github.com/andrewliao11/CoGAN-tensorflow)（NIPS）

* [Invertible Conditional GANs for image editing] [[Paper]](https://drive.google.com/file/d/0B48XS5sLi1OlRkRIbkZWUmdoQmM/view)[[Code]](https://github.com/Guim3/IcGAN)

* [Learning Residual Images for Face Attribute Manipulation] [[Paper]](https://arxiv.org/abs/1612.05363)[[Code]](https://github.com/Zhongdao/FaceAttributeManipulation)(CVPR 2017)

* [Neural Photo Editing with Introspective Adversarial Networks] [[Paper]](https://arxiv.org/abs/1609.07093)[[Code]](https://github.com/ajbrock/Neural-Photo-Editor)(ICLR 2017)

* [Neural Face Editing with Intrinsic Image Disentangling] [[Paper]](https://arxiv.org/abs/1704.04131)(CVPR 2017)

* [GeneGAN: Learning Object Transfiguration and Attribute Subspace from Unpaired Data ] [[Paper]](https://arxiv.org/abs/1705.04932)(BMVC 2017)[[Code]](https://github.com/Prinsphield/GeneGAN)

* [Beyond Face Rotation: Global and Local Perception GAN for Photorealistic and Identity Preserving Frontal View Synthesis] [[Paper]](https://arxiv.org/abs/1704.04086)(ICCV 2017)


### 이산분포(Discrete Distributions)

* [Maximum-Likelihood Augmented Discrete Generative Adversarial Networks] [[Paper]](https://arxiv.org/abs/1702.07983v1)

* [Boundary-Seeking Generative Adversarial Networks] [[Paper]](https://arxiv.org/abs/1702.08431)

* [GANS for Sequences of Discrete Elements with the Gumbel-softmax Distribution] [[Paper]](https://arxiv.org/abs/1611.04051)

### 분류 및 인식 기능 향상

* [Generative OpenMax for Multi-Class Open Set Classification] [[Paper]](https://arxiv.org/pdf/1707.07418.pdf)(BMVC 2017)

* [Controllable Invariance through Adversarial Feature Learning] [[Paper]](https://arxiv.org/abs/1705.11122)[[Code]](https://github.com/github-pengge/adversarial_invariance_feature_learning)(NIPS 2017)

* [Unlabeled Samples Generated by GAN Improve the Person Re-identification Baseline in vitro] [[Paper]](https://arxiv.org/abs/1701.07717)[[Code]](https://github.com/layumi/Person-reID_GAN) (ICCV2017)

* [Learning from Simulated and Unsupervised Images through Adversarial Training] [[Paper]](https://arxiv.org/abs/1612.07828)[[Code]](https://github.com/carpedm20/simulated-unsupervised-tensorflow)（Apple paper, CVPR 2017 Best Paper）

### 프로젝트 

* [cleverhans] [[Code]](https://github.com/openai/cleverhans)(A library for benchmarking vulnerability to adversarial examples)

* [reset-cppn-gan-tensorflow] [[Code]](https://github.com/hardmaru/resnet-cppn-gan-tensorflow)(Using Residual Generative Adversarial Networks and Variational Auto-encoder techniques to produce high-resolution images)

* [HyperGAN] [[Code]](https://github.com/255bits/HyperGAN)(Open source GAN focused on scale and usability)

### Tutorials

* [1] [Ian Goodfellow's GAN Slides](http://www.iangoodfellow.com/slides/2016-12-04-NIPS.pdf) (NIPS Goodfellow Slides)[[Chinese Trans]](http://c.m.163.com/news/a/C7UE2MLT0511AQHO.html?spss=newsapp&spsw=1)[details](https://arxiv.org/pdf/1701.00160v1.pdf)

* [2] [PDF](https://drive.google.com/file/d/0BxKBnD5y2M8NbzBUbXRwUDBZOVU/view)(NIPS Lecun Slides)

* [3] [ICCV 2017 Tutorial About GANS](https://sites.google.com/view/iccv-2017-gans/schedule)