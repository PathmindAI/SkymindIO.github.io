---
title: The MNIST database
short_title: MNIST database
description: An introduction to the MNIST database and its use in machine learning and deep learning.
---

MNIST is a database. The acronym stands for "Modified National Institute of Standards and Technology." The MNIST database contains handwritten digits (0 through 9), and can provide a baseline for testing image processing systems.

![MNIST Hand-written Digits](/images/wiki/mnist-digits.png){:target="_blank"}

MNIST is the "hello world" of machine learning. Data scientists will train an algorithm on the MNIST dataset simply to test a new architecture or framework, to ensure that they work.

MNIST is divided into two datasets: the training set has 60,000 examples of hand-written numerals, and the test set has 10,000. MNIST is a subset of a larger dataset available at the National Institute of Standards and Technology. All of its images are the same size, and within them, the digits are centered and size normalized.

Because MNIST is a labeled dataset that pairs images of hand-written numerals with the name of the respective numeral, it can be used in [supervised learning](./supervised-learning) to train classifiers. It is a good example, alongside Fei Fei Li's ImageNet, of how a good, labeled dataset can advance the cause of machine learning more broadly. More examples of [open datasets are here](open-datasets).

Yann LeCun, now the chief AI scientist at Facebook AI Research (FAIR), wrote a [web page describing MNIST in depth](http://yann.lecun.com/exdb/mnist/). LeCun's worked on image processing in the early 1990s at Bell Labs, with a focus on recognizing hand-written numerals and as part of a larger project to help banks automate check processing.

## MNIST examples

The MNIST examples can be downloaded in four files:

* [training set images (9912422 bytes)](http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz)
* [training set labels (28881 bytes)](http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz)
* [test set images (1648877 bytes)](http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz)
* [test set labels (4542 bytes)](http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz)

### Further Resources

* [Example: Classifying MNIST with Deeplearning4j](https://github.com/deeplearning4j/dl4j-examples/tree/master/dl4j-examples/src/main/java/org/deeplearning4j/examples/feedforward/mnist)
* [Yann LeCun's Website: The MNIST database](http://yann.lecun.com/exdb/mnist/)
