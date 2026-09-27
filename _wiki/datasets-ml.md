---
title: Datasets and Machine Learning
short_title: Datasets and Machine Learning
description: A beginner's reference for generating the right data for machine learning problems.
---

One of the hardest problems to solve in deep learning has nothing to do with neural nets: it's the problem of getting the *right data* in the *right format*.

Getting the right data means gathering or identifying the data that correlates with the outcomes you want to predict; i.e. data that contains a signal about events you care about. The data needs to be aligned with the problem you're trying to solve. Kitten pictures are not very useful when you're building a facial identification system. Verifying that the data is aligned with the problem you seek to solve must be done by a data scientist. If you do not have the right data, then your efforts to build an AI solution must return to the data collection stage.

The right end format for deep learning is generally a tensor, or a multi-dimensional array. So data pipelines built for deep learning will generally convert all data -- be it images, video, sound, voice, text or time series -- into vectors and tensors to which linear algebra operations can be applied. That data frequently needs to be normalized, standardized and cleaned to increase its usefulness, and those are all steps in machine-learning ETL. Deeplearning4j offers the DataVec ETL tool to perform those data preprocessing tasks.

Deep learning, and machine learning more generally, needs a good training set to work properly. Collecting and constructing that body of examples takes time and domain-specific knowledge of where and how to gather relevant information. The network adjusts its parameters using these examples and a training objective, such as predicting their labels or reconstructing their input. We then evaluate how well those learned patterns apply to data the network has not seen.

At this stage, knowledgeable humans need to find the right raw data and transform it into a numerical representation that the deep-learning algorithm can understand, a tensor. Building a training set is, in a sense, pre-pre-training.

Training sets that require much time or expertise can serve as a proprietary edge in the world of data science and problem solving. The nature of the expertise is largely in telling your algorithm what matters to you by selecting what goes into the training set.

It involves telling a story -- through the initial data you select -- that will guide your deep-learning nets as they extract the significant features, both in the training set and in the raw data they've been created to study.

To create a useful training set, you have to understand the problem you're solving; i.e. what you want your deep-learning nets to pay attention to, which outcomes you want to predict.

### The Different Data Sets of Machine Learning

A common setup separates the data into **training**, **validation** and **test** sets.

The **training set** supplies the examples used to fit the model's parameters. The **validation set** helps you choose hyperparameters, such as network size, and decide when to stop training. The **test set** estimates performance after those choices are fixed.

If validation results disappoint, revisit the data or training choices. If you change the model after inspecting test results, that test set has also informed model selection; obtain a fresh held-out evaluation before reporting final performance. Fit preprocessing, including normalization, on training data alone. The [scikit-learn evaluation guide](https://scikit-learn.org/stable/modules/cross_validation.html) explains these roles and cross-validation alternatives.

Choose splits that resemble deployment. Random splits suit independent examples. For forecasts, train on earlier observations and evaluate on later ones. If the goal is to generalize to new people, keep each person's records together when assigning sets.
