---
title: Simulation, AI, Optimization and Complexity
short_title: Simulation, AI and Optimization
description: How simulations model complex systems and help compare optimization methods for supply chains and manufacturing.
---

## What Are Simulations and Why Are They Useful?

Simulation has been called the third scientific method, after empirical experiments and analytical theory. The advantage that computational simulation holds over the other two methods is in how it enables us to explore complexity and draw conclusions about the principles driving complex systems. 

Computer simulations are “what-if machines” that can represent an array of many possible worlds; they help surface emergent phenomena arising from simple rules you devise, such as Conway’s [Game of Life](https://en.wikipedia.org/wiki/Conway%27s_Game_of_Life). 

<!--![Conway game of life](/images/wiki/conway-game-of-life.gif){:target="_blank"}-->

So simulations are tools for thought, and they are most useful when exploring complex and dynamic situations with many components that are in non-linear relationships to each other. Warren Weaver described them as "problems of organized complexity." They can imitate, or recreate, a real-world process or situation.<sup>[1](#one)</sup>

Many people are familiar with simulations that come in the form of games. We have driving simulations, flight simulations, farming simulations, and simulations that allow us to create whole cities, like [SimCity](https://www.theverge.com/2020/5/25/21269612/simcity-maxis-business-simulations-department-history-simenergy-simhealth) or [Cities: Skylines](https://www.citiesskylines.com/). 

![Cities Skylines sim](/images/wiki/cities-skylines.jpg.webp){:target="_blank"}

While most simulations include a visual rendering like the one above, which helps us more deeply grasp how they work, their essence is to show relations between variables, like an algebraic equation does; e.g. `profit = revenue – expenses`. 

Simulating many different revenue and expense levels gives us an idea of how profit might behave under different conditions. In fact, the most common simulation tool is a simple spreadsheet such as Excel. 

So businesses were early creators of simulations to imitate, or model, parts of their operations, which go beyond revenue to include supply chains, logistics, or manufacturing. Those simulations were the basis of predictions and planning; i.e. making optimal decisions. Simulations are not only tools for thought, but tools for action. They allow us to envision how we might shift the simulation's response toward the outcome we desire. 

<!-- Imagine a global supply chain stretching from Vietnam to Denver through ports, transport hubs and warehouses before it provisions a shop on Main Street. Goods move in one direction, money moves in another, and information flows both ways as orders are placed and factory output is calibrated. These are complex systems of inter-related parts.--> 

<a class="w-button button-skilcta" href="https://pathmind.com/" style="width:75%; margin-top: 15px;" target="_blank">Learn How to Apply AI to Simulations >></a>

## Optimization for Simulations

Changes to one part of the system can have an unforeseen impact on another. Sometimes those changes are deliberate improvements, other times they are external shocks that impact operations, such as the widespread supply and demand disruptions occurring during the COVID-19 lockdowns. 

Simulation modelers often seek to explore changes in a system that will help them achieve better outputs to reach their goals; e.g. in business, it would be greater profitability or efficiency, while in public health, it might be greater immunity and fewer deaths. Simulations help them surface causal relationships that they wouldn’t otherwise see.

So people use simulations to explore a space of possibilities; i.e. “What happens if I change the model like this, or that...?” And that exploration will often have the goal of improving operations. This goes beyond the analysis of historical data, which shows what was done in the past, to reveal what might happen if you took an action for the first time, which is only possible because you have modeled the logic of your system in the simulation in order to interact with it. 

A business might seek a better way to configure a supply chain. Its factories have different capacities, orders arrive at different times, and weather can disrupt deliveries. Simulation lets the business test how a proposed configuration behaves over time. An optimizer can use those results to search for configurations that meet an objective, such as reducing delivery delays within a fixed budget.

Optimization tools automate some of that search. A simulation can evaluate candidate settings, such as warehouse reorder thresholds, while a search algorithm proposes settings to try next. [Evolutionary algorithms](./evolutionary-genetic-algorithm) are one option: [differential evolution](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.differential_evolution.html), for example, searches among candidate parameter values without requiring gradients. Simulation packages also offer dedicated optimization engines, such as [AnyLogic's genetic and OptQuest engines](https://anylogic.help/anylogic/experiments/optimization.html). The choice depends on the decisions you can change, the cost of each run, and the constraints a solution must satisfy.

[Deep reinforcement learning](./deep-reinforcement-learning) can learn a policy, a rule for choosing actions, when the problem calls for a sequence of decisions as conditions change. For example, a policy can route each arriving order based on current queues and machine availability. Training uses repeated interaction with the simulation to learn which actions improve the chosen reward. Compare the learned policy with simpler dispatch rules or other optimization methods under the same simulated conditions.

## Why Complexity Matters

Complexity matters because many small components can organize themselves to produce emergent behavior on a level more meaningful to us. 

* Many molecules organize as a cell
* Many cells as a body
* Many bodies as a team

At each scale, the relevant entity is a superorganism composed of the entities one level below. 

When we analyze teams, we see that they behave, and coalesce, in different ways that [impact their performance](https://aeon.co/essays/what-complexity-science-says-about-what-makes-a-winning-team). Many of the behaviors that maximize team performance are difficult to measure and track on the level of the individual; indeed, by incenting individuals to achieve certain personal metrics (in basketball, it might be rebounds), teams can undermine their own goals. Attempting to solve a problem at the wrong level of abstraction only makes it worse. Simulations allow us to observe agents as they organize themselves into a larger operative entity. 

Businesses hire individuals, but output depends on how their work fits together. A factory faces the same coordination problem with machines and robots: speeding up one station can leave another waiting or create a queue. A simulation lets you test scheduling rules against those shared constraints. You might compare a learned dispatch policy with a simple rule that assigns each job to the shortest queue.

## Open-source Simulation Tools

### [Netlogo](https://ccl.northwestern.edu/netlogo/)

NetLogo is a multi-agent programmable modeling environment.

### [Simpy](https://simpy.readthedocs.io/en/latest/)

Simpy is an open-source Python framework for modeling discrete events.

### [PySCeS](http://pysces.sourceforge.net/)

PySCeS provides a variety of tools for the analysis of cellular systems.

### [ManPy](https://www.manpy-simulation.org/)

Discrete event simulation in Python.

### [Project Chrono](https://projectchrono.org/)

An Open Source Multi-physics Simulation Engine.

## Games as Simulation tools

Games have a long history as simulation tools, starting with **[Sim City](https://arstechnica.com/gaming/2015/10/from-simcity-to-well-simcity-the-history-of-city-building-games/)**. 

### [Cities Skylines](https://www.citiesskylines.com/)

Used with modifications to simulate many urban planning and disaster relief efforts. 

### [Unity](https://unity.com/madewith)

Used to simulate smaller scale vehicle movement scenarios, among others. 

## Commercial Simulation Software Tools for Business Operations

The simulation software tools listed below can be used to model business processes. 

<!-- ## Product Design Simulation Tools -->

## Further Reading on the AI Wiki

* [Deep Reinforcement Learning](./deep-reinforcement-learning)
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

## Footnotes

<a name="one">1)</a> *While our focus is on computational simulations, it is interesting to note that novels are a form of simulation built with words. Human imagination is often a simulation built with biological neurons. But we are interested in computer-based models that digitally capture some aspect of a world built with atoms.* 

## Other Resources on Simulations and Complexity

* [Wikipedia: List of Computer Simulation Software](https://en.wikipedia.org/wiki/List_of_computer_simulation_software)
* [Warren Weaver: Science and Complexity](http://people.physics.anu.edu.au/~tas110/Teaching/Lectures/L1/Material/WEAVER1947.pdf)
* [Measures of Complexity: A non-exhaustive list](http://web.mit.edu/esd.83/www/notebook/Complexity.PDF)
* [Santa Fe Institute: Introduction to Complexity](https://www.complexityexplorer.org/)
* [Complexity: The Emerging Science at the Edge of Order and Chaos (1992)](https://www.amazon.com/COMPLEXITY-EMERGING-SCIENCE-ORDER-CHAOS/dp/0671872346)
* [All stars: Is a great team more than the sum of its players? Complexity science reveals the role of strategy, synergy, swarming and more](https://aeon.co/essays/what-complexity-science-says-about-what-makes-a-winning-team)
* [The Success Equation, by Michael Mauboussin](http://success-equation.com/)
* [Fallibility, Reflexivity, and the Human Uncertainty Principle, by George Soros](https://www.georgesoros.com/2014/01/13/fallibility-reflexivity-and-the-human-uncertainty-principle-2/)
* [Non-linear dynamics and chaos, by Steven Strogatz](http://users.uoa.gr/~pjioannou/nonlin/Strogatz,%20S.%20H.%20-%20Nonlinear%20Dynamics%20And%20Chaos.pdf)
