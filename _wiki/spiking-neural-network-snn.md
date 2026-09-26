---
title: Spiking Neural Networks
short_title: Spiking Neural Networks
description: Spiking is a way to encode digital communications over a long distance.
---

Spiking is a way to encode digital communications over a long distance (the spike rate and timing of individual spikes relative to others are the variations by which a spiking signal is encoded), because analog values are destroyed when sent a long distance over an active medium. Think smoke signals in the American West, talking drums in West Africa, or Morse Code on the telegraphs of the 19th and early 20th centuries. 

But analog signals work fine locally, so in a sense, spikes are similar to packets in mesh interconnect. Pure spiking works in all-purpose machines like CPUs and GPUs, but the hardware's numeric capacity is wasted, and it doesn't use scarce random access memory bandwidth optimally.

Like most algorithms, SNNs can be baked onto silicon. When companies like IBM and Intel discuss their "neuromorphic" chips, such as IBM's TrueNorth, they're usually referring to a custom chip, or ASIC, that contains a spiking mechanism in the form of an signal accumulator that fires once a certain type of input surpasses a threshhold. 

Spiking neural networks can learn using [gradient descent](https://arxiv.org/abs/1706.04698), according to research by Dongsung Huh and Terry Sejnowski.

## SNN Advantages

* Low energy usage
* Greater parallelizability due to local-only interactions
* (Maybe) better able to learn non-differentiable functions

## Further Reading on Spiking Neural Networks

* [Gradient Descent for Spiking Neural Networks](https://arxiv.org/abs/1706.04698){:target="_blank"}
* [(A Bit of) Biological Neural Networks – Part I, Spiking Neurons](https://www.fourmilab.ch/cellab/manual/webca.html){:target="_blank"}
* [Spiking Neuron Models. Single Neurons, Populations, Plasticity](http://icwww.epfl.ch/~gerstner/SPNM/SPNM.html){:target="_blank"}
* [STDP-based spiking deep convolutional neural networks for object recognition](https://arxiv.org/abs/1611.01421){:target="_blank"}
* [Convolutional Networks for Fast, Energy-Efficient Neuromorphic Computing](https://arxiv.org/abs/1603.08270){:target="_blank"}
* [TrueHappiness: Neuromorphic Emotion Recognition on TrueNorth](https://arxiv.org/abs/1601.04183){:target="_blank"}
* [The Brain as an Efficient and Robust Adaptive Learner](http://www.cell.com/neuron/abstract/S0896-6273(17)30417-8)
* [Bayesian Spiking Neurons I: Inference](https://www.mitpressjournals.org/doi/abs/10.1162/neco.2008.20.1.91){:target="_blank"}
* [Predictive Coding of Dynamical Variables in Balanced Spiking Networks](http://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1003258){:target="_blank"}
