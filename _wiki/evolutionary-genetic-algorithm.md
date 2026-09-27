---
title: A Beginner's Guide to Genetic & Evolutionary Algorithms
author: Chris V. Nicholson
short_title: Evolutionary & Genetic Algorithms
description: In artificial intelligence, an evolutionary algorithm (EA) is a subset of evolutionary computation, a generic population-based metaheuristic optimization algorithm.
---

> *There is grandeur in this view of life, with its several powers, having been originally breathed into a few forms or into one; and that, whilst this planet has gone cycling on according to the fixed law of gravity, from so simple a beginning endless forms most beautiful and most wonderful have been, and are being, evolved. -Charles Darwin, The Origin of Species* 

Contents

* <a href="#natural">Natural Selection</a>
* <a href="#crossover">Reproduction and Crossover</a>
* <a href="#mutation">Mutation</a>
* <a href="#algo">Genetic and Evolutionary Algorithms</a>
* <a href="#evolution">DeepMind: Combining Neural Networks and Evolutionary Architectures</a>
* <a href="#further">Further Reading</a>

Just as artificial neural networks capture the imagination by comparing algorithms with neurons in an animate brain, genetic algorithms appeal to the metaphor of evolution, nature's most widely known optimization algorithm. 

Genetic and evolutionary algorithms approach mathematical optimization (how do I maximize or minimize a certain value?) in similar ways. What they have in common are ideas drawn from biology: natural selection, reproduction and genetic mutation.<sup>[0](#zero)</sup> 

Let’s do a quick review of the basic mechanisms of biological evolution, which we'll then map to the algorithms.

{% include wiki-inline-cta.html %}

### <a name="natural">Natural Selection</a>

The process of [natural selection](https://www3.beacon-center.org/blog/2012/10/01/evolution-101-natural-selection/) kills living beings that are unfit for their environments, while those more suited to that environment, by definition, survive  and reproduce more prolifically. Thus, those individuals who survived long enough to breed successfully  were “selected” by nature, which imposes a rough filter of predators and toxins on the organisms within it, and cuts many threads short. The value these individuals are maximizing is their number of descendents, or copies of their genes circulating in the gene pool. 

For example, during Britain’s industrial revolution in the mid 19th century, the soot scattered by the new factories' smokestacks darkened the landscape and affected the country’s [peppered moths](http://www.mothscount.org/text/63/peppered_moth_and_natural_selection.html). In areas of greater pollution, such as the cities, the proportion of black-winged moths increased because their wing-darkening genetic mutation left them better camouflaged against the birds that fed on them. The formerly predominant, paler-winged moths remained common only in the less polluted areas of the countryside, where they still blended in and could avoid being dinner. Thus, a change in the environment led to a change in the moth population (fitness is situational...). Darker wings were selected through many small acts of predation, until a new sort of moth emerged and spread. Peppered moths are now called “Darwin’s moth.” 

### <a name="crossover">Reproduction and Crossover</a>

When two animals breed, they mix their genes, and those mixed genes are expressed in the child, a new organism. That child represents a genetic experiment in a sense, a test of the world’s environment that a species conducts ever so slowly, one brood at a time. If the new genetic mix is successful, the child’s genes are propagated to its descendents, and so on to theirs ad infinitum (barring radical changes in the environment), until like [Genghis Khan](http://blogs.discovermagazine.com/gnxp/2010/08/1-in-200-men-direct-descendants-of-genghis-khan/#.XD59F89KjBI) one individual might become father of 1 in every 200 men on the face of the globe. 

In organisms like humans, reproduction mixes the two halves of a diploid chromosome, one contributed by each parent. Crossover, or recombination, is something different that happens when biological organisms produce gametes (sperm, eggs) whose chromosomes will join in the child. The process of creating gametes is called meiosis, and during meiosis, the chromosomes that are being copied and separated during the cell division *swap certain portions of their genes*. Bits of genetic information are exchanged for other bits of genetic information stored on a sister chromosome, and an utterly novel sequence of genes is created, which manifests in a more varied species. The builders of genetic algorithms mimic this process to create variation in the parameters of the algorithms tested, swapping digital bits instead of genetic ones.

### <a name="mutation">Mutation</a>

As genes are copied and relayed from one generation to the next, mutations creep in. The genes’ order might be misread, and one piece of genetic information substituted for another. From one perspective, mutations look like mistakes, and many indeed lead to the [death or impairment of the organism](https://en.wikipedia.org/wiki/Aneuploidy). From another perspective, mutations allow a species to explore the space of all possible genetic combinations, and in so doing, they show whether or not a totally new combination of genes is better than anything that was born before. Mutations ensure diversity, which itself is a hedge that populations make against disease. Mutations also support the continued exploration of a large combinatorial (and undifferentiable) problem, which is especially important given that environments change,<sup>[1](#one)</sup> and those changes can kill off a stagnant and homogeneous species, or favor a novel mutation in its ranks.  

Genetic variation emerges due to damaged DNA, transposition, errors in DNA replication, broken DNA repair processes and recombination; in algorithms, it results from deliberate point mutations in parameters (e.g. random-number generation), as well as crossover.

### <a name="algo">Genetic and Evolutionary Algorithms</a>

Genetic and evolutionary algorithms apply the above ideas to mathematical functions. You could say that a genetic algorithm is like a species. It spawns many singular and unique variations of itself, and those variations are like moth children doomed to be tested against the rigors of the environment. While the environment in real life tests many things about an organism -- strength, intelligence, emotional IQ, fashion sense -- with algorithms, we usually have a single measure of performance: how well an individual instance does with a so-called fitness function.  

Fitness is a measure of how well an algorithm performs against its predictive goal. Let’s say you want to predict house prices based on square footage. Your input is the number of square feet in the house and the output is a dollar amount, the slope of your line is a and the y-axis intercept is b:

`dollar_amount = a * square_feet + b`

![Linear regression](/images/wiki/linear_regression.png)

Different combinations of `a` and `b` are going to give you different lines pointing up, down and sideways, and some of those lines will “fit” the data better than others. Fitness, in this case, minimizes the distance between the data points and the line. Some lines will cut through the scatter plot very far from most of the points (imagine a vertical line through one edge of the dots). The line you see slices the cloud of dots through the center, minimizing the aggregate distance of all dots from the closest points on the line. 

Now let’s say you want a separate, meta-algorithm to find the right function, the fittest slope and intercept for those dots. Since many problems are too complicated to eyeball, you want to computationally seek the best function. That’s where genetic and evolutionary algorithms come in. 

A genetic algorithm would begin by randomly generating a group of linear regression functions, with slopes and intercepts that are clearly unsuited to the data at hand. Those randomly generated lines are then measured against the data to calculate their total error. And the lines with the least error are selected, like survivors snatched from the gladiators’ pit, to create new functions for the next test. Those winning algorithms are recombined; e.g. you might take the slope of one and the intercept of another. And with several of them, you may introduce mutations; i.e. variations on the fittest parameters, increasing or decreasing them with the purpose of testing the mutated variety. 

Spawn, cull, reproduce and mutate: That cycle is repeated until the function surpasses a threshold of fitness acceptable to its author. 

Take it a step further and you can substitute almost any algorithm for linear regression within the testing apparatus of a genetic algorithm, which is really just a search algorithm. For example, you can swap in neural networks, and seek the best structure or hyperparameters for the neural net; i.e. those that allow it to learn the most quickly. 

One of the possible advantages of evolutionary algorithms over neural networks, at least for some problems, is that they do not require gradients; i.e. evolutionary algorithms can explore a parameter space in order to decrease error without depending on [backpropagation](./backpropagation) and differentiation that relates those weights to the error. This is important in environments where reward signals may be sparse and dependencies remote, or when you're dealing with discrete parameters, similar to genes, rather than continuous curves.

## <a name="neural">Combining Neural Networks and Evolutionary Architectures</a>

Recent work by DeepMind to create game-playing AIs has relied on evolutionary ideas as one component in a larger architecture. Specificially, the [AlphaStar II algorithm](https://deepmind.com/blog/article/AlphaStar-Grandmaster-level-in-StarCraft-II-using-multi-agent-reinforcement-learning) was first bootstrapped off human game-players, then allowed to compete with copies of itself, and the most successful versions of the algorithm were cloned and set against each other in what many would call an evolutionary format. 

The central idea combining evolutionary algorithms with neural networks is population-based training. [This paper](https://arxiv.org/abs/1711.09846) provides a good overview of the architecture. It can be applied, not just to neural networks, but also to neural networks embedded in reinforcement learning frameworks. This architecture underpins DeepMind's approach to games. This architecture combines three of the five schools of machine learning described in Pedro Domingos's book The Master Algorithm, excluding only [symbolic reasoning](./symbolic-reasoning) and [Bayesian updates](./bayes-theorem-naive-bayes). 

## <a name="further">Further Reading</a>

* [Nature: Designing neural networks through neuroevolution](https://www.nature.com/articles/s42256-018-0006-z?WT.feed_name=subjects_software)
* [Empiricism and the limits of gradient descent](http://togelius.blogspot.gr/2018/05/empiricism-and-limits-of-gradient.html){:target="_blank"}
* [How Learning Can Guide Evolution, by Hinton and Nowlan (1987)](http://www.cogsci.ucsd.edu/~rik/courses/cogs184_w10/readings/HintonNowlan97.pdf){:target="_blank"}
* [New Methods of Competitive Co-evolution](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.30.3463&rep=rep1&type=pdf){:target="_blank"}
* [The Evolution of Learning: An Experiment in Genetic Connectionism, by Chalmers](http://consc.net/papers/evolution.pdf){:target="_blank"}
* [Deep Neuroevolution: Genetic Algorithms Are a Competitive Alternative for Training Deep Neural Networks for Reinforcement Learning](https://arxiv.org/abs/1712.06567)
* [Distributed Evolutionary Algorithms in Python](https://github.com/DEAP/deap)
* [Evolution Strategies as a Scalable Alternative to Reinforcement Learning](https://blog.openai.com/evolution-strategies/)
* [Population Based Training of Neural Networks](https://arxiv.org/abs/1711.09846)
* [An Introduction to Genetic Algorithms](https://mitpress.mit.edu/books/introduction-genetic-algorithms)

## Other Posts on the Pathmind Wiki

* [Neural Networks and Deep Learning](./neural-network)
* [Recurrent Neural Networks (RNNs) and LSTMs](./lstm)
* [Word2vec and Neural Word Embeddings](./word2vec)
* [Convolutional Neural Networks (CNNs) and Image Processing](./convolutional-network)
* [Accuracy, Precision and Recall](./accuracy-precision-recall-f1)
* [Attention Mechanisms and Transformers](./attention-mechanism-memory-network)
* [Eigenvectors, Eigenvalues, PCA, Covariance and Entropy](./eigenvector)
* [Graph Analytics and Deep Learning](./graph-analysis)
* [Symbolic Reasoning and Machine Learning](./symbolic-reasoning)
* [Markov Chain Monte Carlo, AI and Markov Blankets](./markov-chain-monte-carlo)
* [Generative Adversarial Networks (GANs)](./generative-adversarial-network-gan)
* [AI vs Machine Learning vs Deep Learning](./ai-vs-machine-learning-vs-deep-learning)
* [Multilayer Perceptrons (MLPs)](./multilayer-perceptron)
* [Simulations, Optimization and AI](./simulation-optimization-ai)

<a name="zero">0)</a> *With other machine learning algorithms, it's simple to map their action to that of a human individual, to anthropomorphize them, as it were, and to identify with them. We all embody algorithms in our way, because we're all optimizing for something. But to understand an evolutionary algorithm, you need to identify with an entire species, and imagine the ranks of an entire generation pushing up against the filters of disease and accident and early death. Your body is one member of that generation, and its successful reproduction is the only reward signal that evolution can perceive. That's the only way you enter into history, by submitting a contestant for the next round of breeding and reaping.*

<a name="one">1)</a> *"It is an error to imagine that evolution signifies a constant tendency to increased perfection. That process undoubtedly involves a constant remodeling of the organism in adaption to new conditions; but it depends on the nature of those conditions whether the direction of the modifications effected be up or down." - Thomas Henry Huxley, English Biologist*
