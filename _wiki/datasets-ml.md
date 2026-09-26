---
title: Datasets and Machine Learning
short_title: Datasets and Machine Learning
description: A beginner's reference for generating the right data for machine learning problems.
---

One of the hardest problems to solve in deep learning has nothing to do with neural nets: it's the problem of getting the *right data* in the *right format*.

Getting the right data means gathering or identifying the data that correlates with the outcomes you want to predict; i.e. data that contains a signal about events you care about. The data needs to be aligned with the problem you're trying to solve. Kitten pictures are not very useful when you're building a facial identification system. Verifying that the data is aligned with the problem you seek to solve must be done by a data scientist. If you do not have the right data, then your efforts to build an AI solution must return to the data collection stage.

The right end format for deep learning is generally a tensor, or a multi-dimensional array. So data pipelines built for deep learning will generally convert all data -- be it images, video, sound, voice, text or time series -- into vectors and tensors to which linear algebra operations can be applied. That data frequently needs to be normalized, standardized and cleaned to increase its usefulness, and those are all steps in machine-learning ETL. Deeplearning4j offers the [DataVec ETL tool](./datavec) to perform those data preprocessing tasks.

Deep learning, and machine learning more generally, needs a good training set to work properly. Collecting and constructing the training set -- a sizable body of known data -- takes time and domain-specific knowledge of where and how to gather relevant information. The training set acts as the benchmark against which deep-learning nets are trained. That is what they learn to reconstruct before they're unleashed on data they haven't seen before.

At this stage, knowledgeable humans need to find the right raw data and transform it into a numerical representation that the deep-learning algorithm can understand, a tensor. Building a training set is, in a sense, pre-pre-training.

Training sets that require much time or expertise can serve as a proprietary edge in the world of data science and problem solving. The nature of the expertise is largely in telling your algorithm what matters to you by selecting what goes into the training set.

It involves telling a story -- through the initial data you select -- that will guide your deep-learning nets as they extract the significant features, both in the training set and in the raw data they've been created to study.

To create a useful training set, you have to understand the problem you're solving; i.e. what you want your deep-learning nets to pay attention to, whicn outcomes you want to predict.

### The Different Data Sets of Machine Learning

Machine learning typically works with two data sets: training and test. All three should randomly sample a larger body of data.

The first set you use is the **training set**, the largest of the three. Running a training set through a neural network teaches the net how to weigh different features, adjusting them coefficients according to their likelihood of minimizing errors in your results.

Those coefficients, also known as parameters, will be contained in tensors and together they are called the *model*, because they encode a model of the data they train on. They are the most important takeaways you will obtain from training a neural network.

The second set is your **test set**. It functions as a seal of approval, and you don’t use it until the end. After you’ve trained and optimized your data, you test your neural net against this final random sampling. The results it produces should validate that your net accurately recognizes images, or recognizes them at least [x] percentage of them.

If you don’t get accurate predictions, go back to the training set, look at the hyperparameters you used to tune the network, as well as the quality of your data and look at your pre-processing techniques.

Various [repositories of open data sets](open-datasets) that may be useful in training neural networks are available through the link. 
