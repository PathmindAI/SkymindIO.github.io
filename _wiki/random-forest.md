---
title: Random Forest
short_title: Random Forest
description: A beginner's reference to Random Forests, an ensemble of decision trees.
---

Random forests are made of many decision trees. They are ensembles of decision trees, each decision tree created by using a subset of the attributes used to classify a given population (they are sub-trees, see above). Those decision trees vote on how to classify a given instance of input data, and the random forest bootstraps those votes to choose the best prediction. This is done to prevent overfitting, a common flaw of decision trees. 

A random forest is a supervised classification algorithm. It creates a forest (many decision trees) and orders their nodes and splits randomly. The more trees in the forest, the better the results it can produce.

If you input a training dataset with targets and features into the decision tree, it will formulate some set of rules that can be used to perform predictions. 

Example: You want to predict whether a visitor to your e-commerce Web site will enjoy a mystery novel. First, collect information about past books they've read and liked. Metadata about the novels will be the input; e.g. number of pages, author, publication date, which series it's part of if any. The decision tree contains rules that apply to those features; for example, some readers like very long books and some don't. Inputting metadata about new novels will result in a prediction regarding whether or not the Web site visitor in question would like that novel. Arranging the nodes and defining the rules relies on information gain and Gini-index calculations. With random forests, finding the root node and splitting the feature nodes is done randomly.
