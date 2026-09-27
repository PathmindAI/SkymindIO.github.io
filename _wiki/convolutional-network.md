---
title: A Beginner's Guide to Convolutional Neural Networks (CNNs)
author: Chris V. Nicholson
short_title: Convolutional Neural Network (CNN)
description: A Beginner's Guide to Deep Convolutional Neural Networks (CNNs)
---

Contents

* <a href="#intro">Introduction to Convolutional Neural Networks</a>
* <a href="#tensors">Images, Channels, and Batches</a>
* <a href="#define">Definition of Convolutional Nets</a>
* <a href="#work">How Convolutional Nets Work</a>
* <a href="#max">Maxpooling/Downsampling</a>
* <a href="#code">Just Show Me the Code</a>
* <a href="#resource">More ConvNet Resources</a>

## <a name="intro">Introduction to Deep Convolutional Neural Networks</a>

Convolutional neural networks are neural networks used primarily to classify images (i.e. name what they see), cluster images by similarity (photo search), and perform object recognition within scenes. For example, convolutional neural networks (ConvNets or CNNs) are used to identify faces, individuals, street signs, tumors, platypuses (platypi?) and many other aspects of visual data.

The efficacy of convolutional nets in image recognition is one of the main reasons why the world has woken up to the efficacy of deep learning. In a sense, CNNs are the reason why deep learning is famous. The success of a [deep convolutional architecture called AlexNet](https://en.wikipedia.org/wiki/AlexNet) in the 2012 ImageNet competition was the shot heard round the world. CNNs are powering major advances in computer vision (CV), which has obvious applications for self-driving cars, robotics, drones, security, medical diagnoses, and treatments for the visually impaired.

Convolutional networks can also perform more banal (and more profitable), business-oriented tasks such as optical character recognition (OCR) to digitize text and make natural-language processing possible on analog and hand-written documents, where the images are symbols to be transcribed.

CNNs are not limited to image recognition, however. They have been applied directly to [text analytics](http://www.wildml.com/2015/11/understanding-convolutional-neural-networks-for-nlp/){:target="_blank"}. And they be applied to sound when it is represented visually as a spectrogram, and graph data with [graph convolutional networks](./graph-analysis).

## <a name="tensors">Images, Channels, and Batches</a>

Convolutional neural networks process images as tensors: arrays of numbers arranged along one or more axes.

They can be hard to visualize, so let’s approach them by analogy. A scalar is just a number, such as 7; a vector is a list of numbers (e.g., `[7,8,9]`); and a matrix is a rectangular grid of numbers occupying several rows and columns like a spreadsheet. A matrix has two axes, one for rows and one for columns. Here’s a 2 x 2 matrix:

    [ 1, 2 ]
    [ 5, 8 ]

Stacking matrices adds a third axis. Here’s a 2 x 3 x 2 tensor: two groups, each containing three pairs of numbers. You need three indices to select one number from it.

![tensor](/images/wiki/tensor.png)

In code, the tensor above would appear like this: `[[[2,3],[3,5],[4,7]],[[3,4],[4,6],[5,8]]].` And here's a visual:

![3d matrix cube](/images/wiki/3d_matrix_cube.png)

Arrays can be nested to represent more axes than we can draw in space. The diagram below shows three axes, labeled row, column and page. A fourth axis could identify which stack of pages we want.

![3d matrix](/images/wiki/3d_matrix.png)

With some tools, you will see `NDArray` used synonymously with tensor, or multi-dimensional array. A tensor’s dimensionality `(1,2,3…n)` is called its order; i.e. a fifth-order tensor would have five dimensions.

For one RGB image, the three axes are height, width and color channel. A 30-pixel-high, 30-pixel-wide image has shape `(30, 30, 3)`: one red, one green and one blue value at each pixel. A batch of 32 such images has shape `(32, 30, 30, 3)`. The batch supplies the fourth axis.

A convolution produces *feature maps*, which record responses to learned patterns such as edges or curves. Those maps become the output channels. They occupy the same channel axis that held RGB values in the input. We use the `(batch, height, width, channels)` convention throughout; [Keras also supports placing channels before height and width](https://keras.io/api/layers/convolution_layers/convolution2d/).

{% include wiki-inline-cta.html %}

## <a name="define">Convolutional Definition</a>

From the Latin *convolvere*, "to convolve" means to roll together. For mathematical purposes, a convolution is the integral measuring how much two functions overlap as one passes over the other. Think of a convolution as a way of mixing two functions by multiplying them.

![Convolutional Gaus](/images/wiki/convgaus.gif)

*Credit: [Mathworld](http://mathworld.wolfram.com/). "The green curve shows the convolution of the blue and red curves as a function of t, the position indicated by the vertical green line. The gray region indicates the product `g(tau)f(t-tau)` as a function of t, so its area as a function of t is precisely the convolution."*

Look at the tall, narrow bell curve standing in the middle of a graph. The integral is the area under that curve. Near it is a second bell curve that is shorter and wider, drifting slowly from the left side of the graph to the right. The product of those two functions' overlap at each point along the x-axis is their [convolution](http://mathworld.wolfram.com/Convolution.html). So in a sense, the two functions are being "rolled together."

With image analysis, the static, underlying function (the equivalent of the immobile bell curve) is the input image being analyzed, and the second, mobile function is known as the filter, because it picks up a signal or feature in the image. The two functions relate through multiplication. To visualize convolutions as matrices rather than as bell curves, please see [Andrej Karpathy's excellent animation](https://cs231n.github.io/convolutional-networks/){:target="_blank"} under the heading "Convolution Demo."

The next thing to understand about convolutional nets is that they are passing *many* filters over a single image, each one picking up a different signal. At a fairly early layer, you could imagine them as passing a horizontal line filter, a vertical line filter, and a diagonal line filter to create a map of the edges in the image.

Convolutional networks take those filters, slices of the image's feature space, and map them one by one; that is, they create a map of each place that feature occurs. By learning different portions of a feature space, convolutional nets allow for easily scalable and robust feature engineering.

(In a fully connected RBM, each hidden unit receives input from every visible unit. A convolutional filter instead examines a local patch and uses the same weights at each location.)

So convolutional networks perform a sort of search. Picture a small magnifying glass sliding left to right across a larger image, and recommencing at the left once it reaches the end of one pass (like typewriters do). That moving window is capable recognizing only one thing, say, a short vertical line. Three dark pixels stacked atop one another. It moves that vertical-line-recognizing filter over the actual pixels of the image, looking for matches.

Each time a match is found, it is mapped onto a feature space particular to that visual element. In that space, the location of each vertical line match is recorded, a bit like birdwatchers leave pins in a map to mark where they last saw a great blue heron. A convolutional net runs many, many searches over a single image – horizontal lines, diagonal ones, as many as there are visual elements to be sought.

Convolutional nets perform more operations on input than just convolutions themselves.

The convolution's output is usually passed through a nonlinear activation function. *Tanh* maps finite inputs to values between -1 and 1. The standard *rectified linear unit*, or [ReLU](https://keras.io/api/layers/activation_layers/relu/), computes `max(x, 0)`: negative inputs become zero, and positive inputs pass through unchanged. ReLU has no upper bound.

## <a name="work">How Convolutional Neural Networks Work</a>

The first thing to know about convolutional networks is that they don't perceive images like humans do. Therefore, you are going to have to think in a different way about what an image means as it is fed to and processed by a convolutional network.

{% include wiki-inline-cta.html %}

Convolutional networks perceive images as volumes; i.e. three-dimensional objects, rather than flat canvases to be measured only by width and height. That's because digital color images have a red-green-blue (RGB) encoding, mixing those three colors to produce the color spectrum humans perceive. A convolutional network ingests such images as three separate strata of color stacked one on top of the other.

So a convolutional network receives a normal color image as a rectangular box whose width and height  are measured by the number of pixels along those dimensions, and whose depth is three layers deep, one for each letter in RGB. Those depth layers are referred to as *channels*.

As images move through a convolutional network, we will describe them in terms of input and output volumes, expressing them mathematically as matrices of multiple dimensions in this form: 30x30x3. From layer to layer, their dimensions change for reasons that will be explained below.

You will need to pay close attention to the precise measures of each dimension of the image volume, because they are the foundation of the linear algebra operations used to process images.

Now, for each pixel of an image, the intensity of R, G and B will be expressed by a number, and that number will be an element in one of the three, stacked two-dimensional matrices, which together form the image volume.

Those numbers are the initial, raw, sensory features being fed into the convolutional network, and the ConvNets purpose is to find which of those numbers are significant signals that actually help it classify images more accurately. (Just like other feedforward networks we have discussed.)

Rather than focus on one pixel at a time, a convolutional net applies a *filter*, also called a *kernel*, to a small patch. In an ordinary convolution, each filter spans all input channels. For our RGB image, a filter covering a 3x3 patch therefore has shape `3x3x3`, with a separate set of nine weights for each color channel.

<iframe src="https://cs231n.github.io/assets/conv-demo/index.html" width="100%" height="700px;" style="border:none;"></iframe>
*Credit for this excellent animation goes to [Andrej Karpathy](https://cs231n.github.io/).*

First, look at just one channel of the image and the corresponding slice of the filter. One is 30x30, and the other is 3x3. The filter covers one-hundredth of that channel's surface area.

We are going to take the dot product of the filter with this patch of the image channel. If the two matrices have high values in the same positions, the dot product's output will be high. If they don't, it will be low. In this way, a single value -- the output of the dot product -- can tell us whether the pixel pattern in the underlying image matches the pixel pattern expressed by our filter.

Let's imagine that our filter expresses a horizontal line, with high values along its second row and low values in the first and third rows. Now picture that we start in the upper lefthand corner of the underlying image, and we move the filter across the image step by step until it reaches the upper righthand corner. The size of the step is known as *stride*. You can move the filter to the right one column at a time, or you can choose to make larger steps.

At each position, you take another dot product. Summing these products across the input channels and adding the filter's bias gives one value in the output *feature map*. Its width equals the number of horizontal positions the filter visits. With the same input, filter size and padding, a larger stride produces a smaller map and requires fewer computations.

A 3x3 filter with stride three visits rows 1–3, then 4–6, and so on. With a 30x30 image and no padding, it fits at ten positions along each axis, producing a 10x10 map. For RGB input, its three channel slices produce contributions that are summed at each position. All 27 weights, plus one bias, belong to a single filter that produces one output map.

Now, because images contain many kinds of patterns, we can learn 96 different filters. They produce 96 maps, giving an output shape of `10x10x96` for one image, or `(32, 10, 10, 96)` for our batch of 32. The diagram below uses a smaller example: a 5x5x3 image padded to 7x7x3, two 3x3x3 filters and stride two produce a 3x3x2 output.

![convolutional network labels](/images/wiki/karpathy-convnet-labels.png)

What we just described is a convolution. You can think of Convolution as a fancy kind of multiplication used in signal processing. Another way to think about the two matrices creating a dot product is as two functions. The image is the underlying function, and the filter is the function you roll over it.

![convolutional gaussian](/images/wiki/convgaus.gif)

Large images require considerable storage and computation. A convolution with a larger stride can reduce the height and width of the output, even while the number of channels increases. Pooling offers another way to reduce those spatial dimensions.

## <a name="max">Max Pooling/Downsampling with CNNs</a>

A convolutional layer may be followed by *max pooling*, a form of downsampling. Max pooling takes the largest value in each local window of a feature map. Applied separately to each channel, it can reduce height and width while preserving the number of channels.

![max pooling](/images/wiki/maxpool.png)
*Credit to [Andrej Karpathy](https://cs231n.github.io/){:target="_blank"}.*

For example, 2x2 pooling windows with stride two turn a 10x10x96 volume into a 5x5x96 volume. Each output retains the largest value in its window, losing the other values and the precise position of the maximum within that window.

Much information about lesser values is lost in this step, which has spurred research into alternative methods. But downsampling has the advantage, precisely because information is lost, of decreasing the amount of storage and processing required.

### Alternating Layers

The image below is another attempt to show the sequence of transformations involved in a typical convolutional network.

![convolutional network](/images/wiki/convnet.png)

From left to right you see:

* The actual input image that is scanned for features. The light rectangle is the filter that passes over it.
* Activation maps stacked atop one another, one for each filter you employ. The larger rectangle is one patch to be downsampled.  
* The activation maps condensed through downsampling.
* A new set of activation maps created by passing filters over the first downsampled stack.
* The second downsampling, which condenses the second set of activation maps.
* A fully connected layer that classifies output with one label per node.

As more and more information is lost, the patterns processed by the convolutional net become more abstract and grow more distant from visual patterns we recognize as humans. So forgive yourself, and us, if convolutional networks do not offer easy intuitions as they grow deeper.

## Further Reading

* [Deep Neural Networks](./neural-network)
* [Recurrent Neural Networks (RNNs) and LSTMs](./lstm)
* [Word2vec and Neural Word Embeddings](./word2vec)
* [Accuracy, Precision and Recall](./accuracy-precision-recall-f1)
* [Attention Mechanisms and Transformers](./attention-mechanism-memory-network)
* [Eigenvectors, Eigenvalues, PCA, Covariance and Entropy](./eigenvector)
* [Graph Analytics and Deep Learning](./graph-analysis)
* [Symbolic Reasoning and Machine Learning](./symbolic-reasoning)
* [Markov Chain Monte Carlo, AI and Markov Blankets](./markov-chain-monte-carlo)
* [Deep Reinforcement Learning](./deep-reinforcement-learning)
* [Generative Adversarial Networks (GANs)](./generative-adversarial-network-gan)
* [AI vs Machine Learning vs Deep Learning](./ai-vs-machine-learning-vs-deep-learning)
* [Multilayer Perceptrons (MLPs)](./multilayer-perceptron)
