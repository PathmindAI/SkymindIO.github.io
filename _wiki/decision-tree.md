---
title: Decision Tree
short_title: Decision Tree
description: A decision tree uses a sequence of feature-based questions to predict a category or a numerical value.
---

A decision tree is a series of nodes connected by branches. You can think of it as a flow chart: follow questions about the input from the root node until you reach a prediction at a leaf. A classification tree predicts a category, while a regression tree predicts a numerical value.

![decision tree](/images/wiki/decision_tree.png)

A regression tree might ask about a house's floor area and location, then predict its sale price. With squared-error training, each leaf predicts the mean price of the training houses that reached it. If a leaf contains houses sold for $200,000, $240,000 and $280,000, a new house reaching that leaf receives a $240,000 prediction. Splits are chosen to reduce prediction error. See the [scikit-learn explanation of regression trees](https://scikit-learn.org/stable/modules/tree.html#regression).

Here are some useful terms for describing a decision tree:

* Root Node: A root node is at the beginning of a tree. It represents entire population being analyzed. From the root node, the population is divided according to various features, and those sub-groups are split in turn at each decision node under the root node. 
* Splitting: It is a process of dividing a node into two or more sub-nodes.
* Decision Node: When a sub-node splits into further sub-nodes, it's a decision node.
* Leaf Node or Terminal Node: Nodes that do not split are called leaf or terminal nodes.
* Pruning: Removing the sub-nodes of a parent node is called pruning. A tree is grown through splitting and shrunk through pruning.  
* Branch or Sub-Tree: A sub-section of decision tree is called branch or a sub-tree, just as a portion of a graph is called a sub-graph.
* Parent Node and Child Node: These are relative terms. Any node that falls under another node is a child node or sub-node, and any node which precedes those child nodes is called a parent node. 

![decision tree nodes](/images/wiki/decision_tree_nodes.png)

Decision trees are a popular algorithm for several reasons:

* Explanatory Power: The output of decision trees is interpretable. It can be understood by people without analytical or mathematical backgrounds. It does not require any statistical knowledge to interpret them. 
* Exploratory data analysis: Decision trees can enable analysts to identify significant variables and important relations between two or more variables, helping to surface the signal contained by many input variables. 
* Minimal data cleaning: Because decision trees are resilient to outliers and missing values, they require less data cleaning than some other algorithms. 
* Any data type: Decision trees can make classifications based on both numerical and categorical variables.
* Non-parametric: A decision tree is a non-parametric algorithm, as opposed to neural networks, which process input data transformed into a tensor, via tensor multiplication using large number of coefficients, known as parameters.

<a class="w-button button-skilcta" href="https://pathmind.com" style="width:75%; margin-top: 15px;" target="_blank">Learn to build AI in Simulations >></a>

**Disadvantages**

* Overfitting: A deep tree can fit details of the training data that fail to generalize. Limiting depth and pruning branches can help; choose those settings using validation data.
* Stepwise predictions: A standard regression tree predicts one value per leaf. Its prediction stays constant within each region and can jump at a split. It can model continuous targets, but this stepwise shape can be a poor fit for a smooth trend.
* Heavy feature engineering: The flip side of a decision tree's explanatory power is that it requires heavy feature engineering. When dealing with unstructured data or data with latent factors, this makes decision trees sub-optimal. Neural networks are clearly superior in this regard. 

One weird thing about decision trees (or random forests) is how conceptually simple they are, while in terms of implementation they're non-trivial. How do you find the optimal split/feature based on entropy? Naively implemented, they require something on the order of O(kNlogN) for each split. Multiply that by the number of leaves (2^depth), and multiply that by the number of trees in your forest.

## Further Reading

* [Introduction to Random Forests and Decision Trees](https://victorzhou.com/blog/intro-to-random-forests/)
