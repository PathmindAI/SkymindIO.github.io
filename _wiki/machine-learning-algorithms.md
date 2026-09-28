---
title: Machine Learning Algorithms
short_title: Machine Learning Algorithms
description: A beginner's reference for algorithm's used in machine learning.
---

## Some Basic Machine Learning Algorithms

Below you'll find descriptions of and links to some basic and powerful machine-learning algorithms, including:

* [Attention Mechanisms & Memory Networks](attention-mechanism-memory-network)
* [Bayes Theorem & Naive Bayes Classifiers](bayes-theorem-naive-bayes)
* [Decision Trees](decision-tree)
* [Eigenvectors, Eigenvalues and Machine Learning](eigenvector)
* [Evolutionary & Genetic Algorithms](evolutionary-genetic-algorithm)
* [Expert Systems/Rules Engines/Symbolic Reasoning](symbolic-reasoning)
* [Generative Adversarial Networks (GANs)](generative-adversarial-network-gan)
* [Graph Analytics and ML](graph-analysis)
* <a href="#linear">Linear Regression</a>
* [Logistic Regression](logistic-regression)
* [LSTMs and Recurrent Neural Networks](lstm)
* [Markov Chain Monte Carlo Methods (MCMC)](markov-chain-monte-carlo)
* [Neural Networks](neural-network)
* <a href="#random">Random Forests</a>
* [Reinforcement Learning](deep-reinforcement-learning)
* [Word2vec, Neural Embeddings and NLP](word2vec)

Machine learning algorithms are programs (math and logic) that adjust themselves to perform better as they are exposed to more data. The "learning" part of machine learning means that those programs change how they process data over time, much as humans change how they process data by learning. So a machine-learning algorithm is a program with a specific way to adjusting its own parameters, given feedback on its previous performance in making predictions about a dataset. 

## <a name="linear">Linear Regression</a>

Linear regression is simple, which makes it a great place to start thinking about algorithms more generally. Here it is:

```
ŷ = a * x + b
```

Read aloud, you'd say "y-hat equals a times x plus b."

* y-hat is the output, or guess made by the algorithm, the dependent variable.
* a is the coefficient. It's also the slope of the line that expresses the relationship between x and y-hat.
* x is the input, the given or independent variable.
* b is the intercept, where the line crosses the y axis.

Linear regression expresses a linear relationship between the input x and the output y; that is, for every change in x, y-hat will change by the same amount no matter how far along the line you are. The x is transformed by the same a and b at every point.

Linear regression with only one input variable is called Simple Linear Regression. With more than one input variable, it is called Multiple Linear Regression. An example of Simple Linear Regression would be attempting to predict a house price based on the square footage of the house and nothing more.

```
house_price_estimate = a * square_footage + b
```
Multiple Linear Regression would take other variables into account, such as the distance between the house and a good public school, the age of the house, etc.  

To fit a line to a scatter plot, compare its prediction `ŷ` with each observed `y`. The difference `y - ŷ` is called a residual. Ordinary least squares chooses `a` and `b` to minimize the sum of **squared residuals**:

```
sum_of_squared_residuals = Σ (y - ŷ)²
```

Residuals of `+3` and `-3` add to `0`, even though both predictions miss by 3. Their squares add to `3² + (-3)² = 9 + 9 = 18`. Squaring prevents opposite errors from canceling and gives larger misses more weight. See [scikit-learn's least-squares explanation](https://scikit-learn.org/stable/modules/linear_model.html#ordinary-least-squares).

That scatter plot of data points may look like a baguette -- long in one direction and short in another -- in which case linear regression may achieve a fit. (If the data points look like a meandering river, a straight line is probably not the right function to use to make predictions.)

![scatter plot](/images/wiki/scatterplot.png)

<a class="w-button button-skilcta" href="https://www.pathmind.com" style="width:75%; margin-top: 15px;" target="_blank">Apply AI to Business Simulations >></a>

## Logistic Regression

[Binary logistic regression](logistic-regression) turns a weighted input score into an estimated class probability with an S-shaped sigmoid function. A decision threshold converts that probability into a class prediction, like an on-off switch controlled by a smooth signal.

## Decision Tree

A [decision tree](decision-tree) asks a sequence of questions about an input's features, like a game of 20 questions. Each answer sends the input down a branch until it reaches a leaf, which supplies a prediction. Classification trees predict categories; regression trees predict numerical values such as house prices. A regression tree trained with squared error predicts the mean target value of the training examples in each leaf.

## <a name="random">Random Forest</a>

Random forests combine many [decision trees](decision-tree) for classification or regression. Each tree typically learns from a bootstrap sample: training rows drawn with replacement, so some rows appear more than once. At each split, the tree considers a fresh random subset of features, or all features if configured to do so. Within that candidate set, the tree chooses the split that best improves its training criterion, such as squared error for regression.

For classification, trees can vote for a class; scikit-learn averages their predicted class probabilities and chooses the class with the highest average. For regression, the forest averages the trees' numerical predictions. Combining diverse trees usually reduces the variation in predictions caused by changes in the training data, which can reduce overfitting. More trees increase computation, and improvements eventually level off. See the [scikit-learn random-forest guide](https://scikit-learn.org/stable/modules/ensemble.html#random-forests).
