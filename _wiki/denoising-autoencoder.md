---
title: Denoising Autoencoders
short_title: Denoising Autoencoders
description: Denoising autoencoders attempt to address identity-function risk by randomly corrupting input (i.e. introducing noise) that the autoencoder must then reconstruct.
---

An autoencoder is a neural network used for dimensionality reduction; that is, for feature selection and extraction. Autoencoders with more hidden layers than inputs run the risk of learning the [identity function](https://en.wikipedia.org/wiki/Identity_function){:target="_blank"} -- where the output simply equals the input -- thereby becoming useless.

Denoising autoencoders are an extension of the basic autoencoder, and represent a stochastic version of it. Denoising autoencoders attempt to address identity-function risk by randomly corrupting input (i.e. introducing noise) that the autoencoder must then reconstruct, or denoise.

### Parameters and Corruption level

The amount of noise to apply to the input takes the form of a percentage. Typically, 30 percent, or 0.3, is fine, but if you have very little data, you may want to consider adding more.

<!-- ### Input/initiating a denoising autoencoder

Setting up a single-thread denoising autoencoder is easy.

To create the machine, you simply instantiate an AutoEncoder and set the corruptionLevel, or noise, as you can see in the example below.

<script src="https://gist-it.appspot.com/https://github.com/deeplearning4j/dl4j-examples/blob/master/src/main/java/org/deeplearning4j/examples/autoencoder/StackedAutoEncoderMnistExample.java?slice=24:96"></script>

That's how you set up a denoising autoencoder with one visible layer and one hidden layer using MNIST data. This net has a learning rate of 0.1, momentum of of 0.9, and utilizes reconstruction cross entropy as its loss function.  -->

### Stacked Denoising Autoencoder

A stacked denoising autoencoder is simply many denoising autoencoders strung together.

It is to a denoising autoencoder what a deep-belief network is to a restricted Boltzmann machine.

A key function of SDAs, and deep learning more generally, is unsupervised pre-training, layer by layer, as input is fed through. Once each layer is pre-trained to conduct feature selection and extraction on the input from the preceding layer, a second stage of supervised fine-tuning can follow.

A word on stochastic corruption in SDAs: Denoising autoencoders shuffle data around and learn about that data by attempting to reconstruct it. The act of shuffling is the noise, and the job of the network is to recognize the features within the noise that will allow it to classify the input. When a network is being trained, it generates a model, and measures the distance between that model and the benchmark through a loss function. Its attempts to minimize the loss function involve resampling the shuffled inputs and re-reconstructing the data, until it finds those inputs which bring its model closest to what it has been told is true.

The serial resamplings are based on a generative model to randomly provide data to be processed. This is known as a Markov Chain, and more specifically, a Markov Chain Monte Carlo algorithm that steps through the data set seeking a representative sampling of indicators that can be used to construct more and more complex features.
