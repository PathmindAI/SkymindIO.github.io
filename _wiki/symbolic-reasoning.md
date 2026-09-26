---
title: Symbolic Reasoning (Symbolic AI) and Machine Learning
author: Chris V. Nicholson
short_title: Symbolic Reasoning
description: Because deep learning models largely lack interpretability, symbolic learning enables models to cohabitate with human readable concepts.
---

Deep learning has [its discontents](https://twitter.com/GaryMarcus/status/1071438612690960384), and many of them look to other branches of AI when they hope for the future. Symbolic reasoning is one of those branches. 

The two biggest flaws of deep learning are its lack of model interpretability (i.e. why did my model make that prediction?) and the large amount of data that deep neural networks require in order to learn. Neural nets are data hungry.

Geoff Hinton himself has [expressed scepticism](https://www.axios.com/artificial-intelligence-pioneer-says-we-need-to-start-over-1513305524-f619efbd-9db0-4947-a9b2-7a4c310a28fe.html){:target="_blank"} about whether backpropagation, the workhorse of deep neural nets, will be the way forward for AI.<sup>[1](#one)</sup>   

Research into so-called one-shot learning may address deep learning's data hunger, while deep symbolic learning, or enabling deep neural networks to manipulate, generate and otherwise cohabitate with concepts expressed in strings of characters, could help solve explainability, because, after all, humans communicate with signs and symbols, and that is what we desire from machines.<sup>[2](#two)</sup> [Recent work](https://deepmind.com/research/publications/neuro-symbolic-concept-learner-interpreting-scenes-words-and-sentences-natural-supervision/) by MIT, DeepMind and IBM has shown the power of combining connectionist techniques like deep neural networks with symbolic reasoning. 

## Signs, Symbols, Signifiers and Signifieds

The words *sign* and *symbol* derive from Latin and Greek words, respectively, that mean *mark* or *token*, as in "take this rose as a token of my esteem." Both words mean "to stand for something else" or "to represent something else". 

That something else could be a physical object, an idea, an event, you name it. For our purposes, the sign or symbol is a visual pattern, say a character or string of characters, in which meaning is embedded, and that sign or symbol is pointing at something else. It could be the variable `x`, pointing at an unknown quantity, or it could be the word `rose`, which is pointing at the red, curling petals layered one over the other in a tight spiral at the end of a stalk of thorns.<sup>[3](#three)</sup>  

The signifier indicates the signified, like a finger pointing at the moon.<sup>[4](#four)</sup>  Symbols compress sensory data in a way that enables humans, large primates of limited bandwidth, to share information with each other.<sup>[5](#five)</sup> You could say that they are necessary to overcome biological chokepoints in throughput. Insofar as computers suffered from the same chokepoints, their builders relied on all-too-human hacks like symbols to sidestep the limits to processing, storage and I/O. As computational capacities grow, the way we digitize and process our analog reality can also expand, until we are juggling billion-parameter tensors instead of seven-character strings. 

Symbols also serve to transfer learning in another sense, not from one human to another, but from one situation to another, over the course of a single individual's life. That is, a symbol offers a level of abstraction above the concrete and granular details of our sensory experience, an abstraction that allows us to transfer what we've learned in one place to a problem we may encounter somewhere else. Given that reward signals are sparse in real life, and difficult to connect to their causes (some of the reasons you're unhappy may have to do with actions you took years ago -- can you guess which ones?), symbols are a way of tranferring reward signals learned in one situation, when confronting another scenario without clear rewards. In a certain sense, every abstract category, like `chair`, asserts an analogy between all the disparate objects called chairs, and we transfer our knowledge about one chair to another with the help of the symbol. 

Combinations of symbols that express their interrelations could be called *reasoning*, and when we humans string a bunch of signs together to express thought, as I am doing now, you might call it *symbolic manipulation*. Sometimes those symbolic relations are necessary and deductive, as with the formulas of pure math or the conclusions you might draw from a logical syllogism like this old Roman chestnut:

```
All men are mortal; Caius is a man; therefore Caius is mortal.
```

Other times the symbols express lessons we derive inductively from our experiences of the world, as in: "the baby seems to prefer the pea-flavored goop (so for godssake let's make sure we keep some in the fridge)," or *E = mc<sup>2</sup>*.  

## Symbolic AI

[Symbolic artificial intelligence](https://en.wikipedia.org/wiki/Symbolic_artificial_intelligence){:target="_blank"}, also known as Good, Old-Fashioned AI (GOFAI), was the dominant paradigm in the AI community from the post-War era until the late 1980s. 

Implementations of symbolic reasoning are called rules engines or expert systems or [knowledge graphs](./graph-analysis). See [Cyc](https://en.wikipedia.org/wiki/Cyc){:target="_blank"} for one of the longer-running examples. Google made a big one, too, which is what provides the information in the top box under your query when you search for something easy like [the capital of Germany](https://www.google.com/search?q=capital+of+germany){:target="_blank"}. These systems are essentially piles of nested if-then statements drawing conclusions about entities (human-readable concepts) and their relations (expressed in well understood semantics like `X` is-a `man` or `X` lives-in `Acapulco`). 

Imagine how Turbotax manages to reflect the US tax code -- you tell it how much you earned and how many dependents you have and other contingencies, and it computes the tax you owe by law -- that's an expert system. 

External concepts are added to the system by its programmer-creators, and that's more important than it sounds... 

One of the main differences between machine learning and traditional symbolic reasoning is where the learning happens. In machine- and deep-learning, the algorithm learns rules as it establishes correlations between inputs and outputs. In symbolic reasoning, the rules are created through human intervention. That is, to build a symbolic reasoning system, first humans must learn the rules by which two phenomena relate, and then hard-code those relationships into a static program. This difference is the subject of a well-known [hacker koan](https://simple.wikipedia.org/wiki/Hacker_koan){:target="_blank"}:

```
In the days when Sussman was a novice, Minsky once came to him as he sat hacking at the PDP-6.
"What are you doing?", asked Minsky.
"I am training a randomly wired neural net to play Tic-tac-toe", Sussman replied.
"Why is the net wired randomly?", asked Minsky.
"I do not want it to have any preconceptions of how to play", Sussman said.
Minsky then shut his eyes.
"Why do you close your eyes?" Sussman asked his teacher.
"So that the room will be empty."
At that moment, Sussman was enlightened.
```

A hard-coded rule is a preconception. It is one form of assumption, and a strong one, while deep neural architectures contain other assumptions, usually about *how* they should learn, rather than what conclusion they should reach. The ideal, obviously, is to choose assumptions that allow a system to learn flexibly and produce accurate decisions about their inputs. 

## Problems with Symbolic AI (GOFAI)

One of the main stumbling blocks of symbolic AI, or GOFAI, was the difficulty of revising beliefs once they were encoded in a rules engine. Expert systems are monotonic; that is, the more rules you add, the more knowledge is encoded in the system, but additional rules can't undo old knowledge. *Monotonic* basically means *one direction*; i.e. when one thing goes up, another thing goes up. Because machine learning algorithms can be retrained on new data, and will revise their parameters based on that new data, they are better at encoding tentative knowledge that can be retracted later if necessary; i.e. if they need to learn something new, like when data is non-stationary. 

A second flaw in symbolic reasoning is that the computer itself doesn't know what the symbols mean; i.e. they are not necessarily linked to any other representations of the world in a non-symbolic way. Again, this stands in contrast to neural nets, which can link symbols to vectorized representations of the data, which are in turn just translations of raw sensory data. So the main challenge, when we think about GOFAI and neural nets, is how to ground symbols, or relate them to other forms of meaning that would allow computers to map the changing raw sensations of the world to symbols and then reason about them.

One question that logically arises then is: who are the symbols for? Are they useful to machines at all? If symbols allow homo sapiens to share and manipulate information based on fundamental physiological constraints, great, but why should machines use them? Why shouldn't machines just talk to each other in vectors or some squeaky language of dolphins and fax machines? Let's hazard a bet: When machines do begin to speak to one another intelligibly, it will be in a language that humans cannot understand. Maybe words are too low-bandwidth for high-bandwidth machines. Maybe they need more dimensions to express themselves unambiguously. Language is just a keyhole in a door that machines have bypassed.<sup>[6](#six)</sup> At best, natural language could be an API that AI offers humans so they can ride on its coattails; at worst, it could be a distraction from what constitutes true machine intelligence. But we have confused it with the summit of achievement, because natural language is how we show that we're smart. 

## Combining Deep Neural Nets and Symbolic Reasoning

How can we fuse the ability of deep neural nets to learn probabilistic correlations from scratch alongside abstract and higher-order concepts, which are useful in compressing data and combining it in new ways? How can we learn to attach new meanings to concepts, and to use [atomic concepts as elements in more complex and composable thoughts](https://medium.com/@GaryMarcus/in-defense-of-skepticism-about-deep-learning-6e8bfd5ae0f1){:target="_blank"} such as language allows us to express in all its natural plasticity? 

Combining symbolic reasoning with deep neural networks and deep reinforcement learning may help us address the fundamental challenges of reasoning, hierarchical representations, transfer learning, robustness in the face of adversarial examples, and interpretability (or explanatory power).

### Supervised Learning: A Basic Hybrid AI

Let's explore how they currently overlap and how they might. First of all, every deep neural net trained by supervised learning combines deep learning and symbolic manipulation, at least in a rudimentary sense. Because symbolic reasoning encodes knowledge in symbols and strings of characters. In supervised learning, those strings of characters are called labels, the categories by which we classify input data using a statistical model. The output of a classifier (let's say we're dealing with an image recognition algorithm that tells us whether we're looking at a pedestrian, a stop sign, a traffic lane line or a moving semi-truck), can trigger business logic that reacts to each classification. That business logic is one form of symbolic reasoning. 

## <a name="footnote">Footnotes</a>

<a name="one">1)</a> *Hinton, Yann LeCun and Andrew Ng have all suggested that work on unsupervised learning (learning from unlabeled data) will lead to our next breakthroughs.*

<a name="two">2)</a> *The two problems may overlap, and solving one could lead to solving the other, since a concept that helps explain a model will also help it recognize certain patterns in data using fewer examples.* 

<a name="three">3)</a> *The weird thing about writing about signs, of course, is that in the confines of a text, we're just using one set of signs to describe another in the hopes that the reader will respond to the sensory evocation and supply the necessary analog memories of `red` and `thorn`. But you get my drift. (It gets even weirder when you consider that the sensory data perceived by our minds, and to which signs refer, are themselves signs of the thing in itself, which we cannot know.)*

<a name="four">4)</a> *In Japanese Buddhism, Zen masters often say that their teachings are like fingers pointing at the moon. The finger is not the moon, but it is directionally useful. So, too, each sign is a finger pointing at sensations.*

<a name="five">5)</a> *According to science, the average American English speaker speaks at a rate of about 110–150 words per minute (wpm). Just how much reality do you think will fit into a ten-minute transmission?*

<a name="six">6)</a> *"All right, now we’re coming to what I promised and led you through the whole dull synopsis of what led up to this in hopes of. Meaning what it’s like to die, what happens. Right? This is what everyone wants to know. And you do, trust me. Whether you decide to go through with it or not, whether I somehow talk you out of it the way you think I’m going to try to do or not. It’s not what anyone thinks, for one thing. The truth is you already know what it’s like. You already know the difference between the size and speed of everything that flashes through you and the tiny inadequate bit of it all you can ever let anyone know. As though inside you is this enormous room full of what seems like everything in the whole universe at one time or another and yet the only parts that get out have to somehow squeeze out through one of those tiny keyholes you see under the knob in older doors. As if we are all trying to see each other through these tiny keyholes. 
But it does have a knob, the door can open. But not in the way you think. But what if you could? Think for a second — what if all the infinitely dense and shifting worlds of stuff inside you every moment of your life turned out now to be somehow fully open and expressible afterward, after what you think of as you has died, because what if afterward now each moment itself is an infinite sea or span or passage of time in which to express it or convey it, and you don’t even need any organized English, you can as they say open the door and be in anyone else’s room in all your own multiform forms and ideas and facets? Because listen — we don’t have much time, here’s where Lily Cache slopes slightly down and the banks start getting steep, and you can just make out the outlines of the unlit sign for the farmstand that’s never open anymore, the last sign before the bridge — so listen: What exactly do you think you are? The millions and trillions of thoughts, memories, juxtapositions — even crazy ones like this, you’re thinking — that flash through your head and disappear? Some sum or remainder of these? Your history? Do you know how long it’s been since I told you I was a fraud? Do you remember you were looking at the respicem watch hanging from the rearview and seeing the time, 9:17? What are you looking at right now? Coincidence? What if no time has passed at all? The truth is you’ve already heard this. That this is what it’s like. That it’s what makes room for the universes inside you, all the endless in-bent fractals of connection and symphonies of different voices, the infinities you can never show another soul. And you think it makes you a fraud, the tiny fraction anyone else ever sees? Of course you’re a fraud, of course what people see is never you. And of course you know this, and of course you try to manage what part they see if you know it’s only a part. Who wouldn’t? It’s called free will, Sherlock. But at the same time it’s why it feels so good to break down and cry in front of others, or to laugh, or speak in tongues, or chant in Bengali — it’s not English anymore, it’s not getting squeezed through any hole." - [David Foster Wallace, "Good Old Neon"](http://sdavidmiller.com/octo/files/no_google2/GoodOldNeon.pdf)* 

## Further Reading on Symbolic AI

* [Logical vs. Analogical or Symbolic vs. Connectionist or Neat vs. Scruffy, by Marvin Minsky](http://web.media.mit.edu/~minsky/papers/SymbolicVs.Connectionist.html){:target="_blank"}
* [Answer to: What was GOFAI, and why did it fail?](https://www.reddit.com/r/artificial/comments/ziw60/what_was_gofai_and_why_did_it_fail/c6531lf/){:target="_blank"}
* McDermott, D. (1987), [A critique of pure reason](http://onlinelibrary.wiley.com/doi/10.1111/j.1467-8640.1987.tb00183.x/full){:target="_blank"}. Computational Intelligence, 3: 151–160. doi: 10.1111/j.1467-8640.1987.tb00183.x 
* [The Symbol Grounding Problem](http://cogprints.org/3106/){:target="_blank"}, Harnad, Stevan (1990) 
* [Intentionality](http://plato.stanford.edu/entries/intentionality){:target="_blank"}
* [Belief Revision](https://en.wikipedia.org/wiki/Belief_revision){:target="_blank"}
* [Non-monotonic Logic](https://en.wikipedia.org/wiki/Non-monotonic_logic){:target="_blank"}
* [The Yale shooting problem](https://en.wikipedia.org/wiki/Yale_shooting_problem){:target="_blank"}
* [ALMECOM: Active Logic, MEtacognitive COmputation, and Mind](http://www.cs.umd.edu/active/){:target="_blank"}
* [SATNet: Bridging deep learning and logical reasoning using a differentiable satisfiability solver](https://arxiv.org/abs/1905.12149)

## Other Posts on the Pathmind Wiki

* [Deep Neural Networks](./neural-network)
* [Recurrent Neural Networks (RNNs) and LSTMs](./lstm)
* [Word2vec and Neural Word Embeddings](./word2vec)
* [Convolutional Neural Networks (CNNs) and Image Processing](./convolutional-network)
* [Accuracy, Precision and Recall](./accuracy-precision-recall-f1)
* [Attention Mechanisms and Transformers](./attention-mechanism-memory-network)
* [Eigenvectors, Eigenvalues, PCA, Covariance and Entropy](./eigenvector)
* [Graph Analytics and Deep Learning](./graph-analysis)
* [Markov Chain Monte Carlo, AI and Markov Blankets](./markov-chain-monte-carlo)
* [Deep Reinforcement Learning](./deep-reinforcement-learning)
* [Generative Adversarial Networks (GANs)](./generative-adversarial-network-gan)
* [AI vs Machine Learning vs Deep Learning](./ai-vs-machine-learning-vs-deep-learning)
* [Multilayer Perceptrons (MLPs)](./multilayer-perceptron)

