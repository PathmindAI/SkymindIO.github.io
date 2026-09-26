---
title: A Beginner's Guide to Differentiable Programming
short_title: Differentiable Programming
description: A new kind of software by assembling networks of parameterized functional blocks and by training them from examples using some form of gradient-based optimization.
---

Differentiable programs are programs that rewrite themselves at least one component by optimizing along a gradient, like neural networks do using optimization algorithms such as gradient descent. Here's a graphic illustrating the difference between differential and probabilistic programming approaches. 

![Differentiable programming](/images/wiki/differentiable_probabilistic.jpg)
*Credit: [Breandon](https://twitter.com/breandan)*

Yann LeCun described [differentiable programming](https://www.facebook.com/yann.lecun/posts/10155003011462143){:target="_blank"} like this:

"*Yeah, Differentiable Programming is little more than a rebranding of the modern collection Deep Learning techniques, the same way Deep Learning was a rebranding of the modern incarnations of neural nets with more than two layers.
The important point is that people are now building a new kind of software by assembling networks of parameterized functional blocks and by training them from examples using some form of gradient-based optimization….It’s really very much like a regular program, except it’s parameterized, automatically differentiated, and trainable/optimizable.
An increasingly large number of people are defining the networks procedurally in a data-dependent way (with loops and conditionals), allowing them to change dynamically as a function of the input data fed to them. It's really very much like a regular progam, except it's parameterized, automatically differentiated, and trainable/optimizable. Dynamic networks have become increasingly popular (particularly for NLP), thanks to deep learning frameworks that can handle them such as PyTorch and Chainer (note: our old deep learning framework Lush could handle a particular kind of dynamic nets called Graph Transformer Networks, back in 1994. It was needed for text recognition).
People are now actively working on compilers for imperative differentiable programming languages. This is a very exciting avenue for the development of learning-based AI.
Important note: this won't be sufficient to take us to "true" AI. Other concepts will be needed for that, such as what I used to call predictive learning and now decided to call Imputative Learning. More on this later....*"

Differentiable programming will never be the buzzword that deep learning is. It's too much of a mouthful. LeCun has proposed DP, DiffProg or dProg as less unwieldy substitutes.

In a more recent [Reddit AMA](https://www.reddit.com/r/science/comments/7yegux/aaas_ama_hi_were_researchers_from_google/){:target="_blank"}, LeCun went on to say:


"*With the ability to define dynamic deep architectures (i.e. computation graphs that are defined procedurally and whose structure changes for every new input) is a generalization of deep learning that some have called Differentiable Programming.*"

Frankly, dynamic computation graphs for deep neural networks sounds an awful lot like a kind of deep learning.

### See Also

* [Deluca: A Differentiable Control Library: Environments, Methods, and Benchmarking](https://arxiv.org/abs/2102.09968)
* [Deluca's GitHub repo](https://github.com/google/deluca)
* [Software 2.0, by Andrej Karpathy](https://medium.com/@karpathy/software-2-0-a64152b37c35){:target="_blank"}
* [Slides: Differentiable Programming, Microsoft Research (2016)](http://www.cs.nuim.ie/~gunes/files/Baydin-MSR-Slides-20160201.pdf){:target="_blank"}
* [Differentiable Programming @Edge.org](https://www.edge.org/response-detail/26794){:target="_blank"}
* [What Is Differentiable Programming?](https://fluxml.ai/2019/02/07/what-is-differentiable-programming.html)
