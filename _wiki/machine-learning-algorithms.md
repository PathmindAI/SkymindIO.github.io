---
title: Choosing a Starting Machine Learning Model
short_title: Machine Learning Algorithms
description: Choose a starting model for your data, establish a baseline, and compare alternatives on validation data.
---

A starting model should be quick to train and give you a result you can compare. Choose it from the inputs you have and the output you need, then measure whether a more complex model improves the result enough to justify its cost.

<a id="some-basic-machine-learning-algorithms"></a>

## Choose by the Output You Need

For data arranged in rows and columns, use these as first experiments:

| Your task | Starting model | What to check |
| --- | --- | --- |
| <span id="linear"></span><span id="linear-regression"></span>Predict a number, such as a sale price | [Ridge regression](https://scikit-learn.org/stable/modules/linear_model.html#ridge-regression-and-classification): a linear model that penalizes large coefficients | Whether prediction errors reveal curves or combinations of features the model misses. |
| <span id="logistic-regression"></span>Predict a category, such as whether an order will be returned | [Logistic regression](./logistic-regression) with a coefficient penalty, to estimate class probabilities | Which mistakes matter and where to set the threshold for a yes/no decision. |
| Find groups without supplied labels | [K-means](./unsupervised-learning#k-means) on numerical features scaled for a meaningful distance measure | Whether compact groups are plausible and whether the resulting clusters are useful. |

The output determines whether a prediction task is regression or classification. A category such as neighborhood can be an input to a model that predicts a numerical sale price. For the linear models above, encode unordered categories with 0-or-1 indicator columns and put numerical inputs on comparable scales.

Linear regression fits a weighted sum of the inputs. Ordinary least squares minimizes squared errors: errors of `+3` and `−3` sum to zero but contribute `18` in squared error. Ridge adds a coefficient penalty to that objective.

<a id="decision-tree"></a>
<a id="random-forest"></a>
<a id="random"></a>

For either numerical or categorical targets, compare the linear model with a [random forest](https://scikit-learn.org/stable/modules/ensemble.html#random-forests) when combinations of features may matter. A forest combines many trees' predictions. If you need a short sequence of questions you can inspect, try a shallow [decision tree](./decision-tree) and measure the accuracy you give up or gain.

## Adapt to Text, Images or Forecasts

**Text classification:** Start with [TF-IDF word features](./bagofwords-tf-idf) and logistic regression. This is inexpensive to train and gives a benchmark for testing whether pretrained text representations improve the distinctions your task needs. See the [scikit-learn text-classification comparison](https://scikit-learn.org/stable/auto_examples/text/plot_document_classification_20newsgroups.html).

**Image classification:** Use a pretrained image network to extract features, keep its weights fixed, and train a new classifier on those features. Then compare with fine-tuning part of the network on your images. The [transfer-learning tutorial](https://www.tensorflow.org/tutorials/images/transfer_learning) demonstrates both approaches.

**Forecasting:** First predict the last observed value, or repeat the corresponding value from the previous season. Compare a model using past observations as inputs against that baseline. Evaluate on later periods, at the forecast horizon you need; [this forecasting example](https://scikit-learn.org/stable/auto_examples/applications/plot_time_series_lagged_features.html) shows why randomly shuffled splits can exaggerate performance.

## Decide Whether to Keep It

For supervised tasks, measure a [simple baseline](https://scikit-learn.org/stable/modules/model_evaluation.html#dummy-estimators) that ignores the inputs. Predict the training mean for squared-error evaluation, the training median for absolute error, or the most common training class for classification accuracy. If a model cannot beat that baseline on validation data, inspect the inputs and labels before adding complexity.

Choose an evaluation measure that matches the cost of mistakes. Mean absolute error reports a numerical prediction's average miss in the target's units. For classification, examine [precision and recall](./accuracy-precision-recall-f1) when false alarms and missed cases have different consequences.

Compare candidates on the same validation splits. For predictions about new customers, keep each customer's records in one split. Fit preprocessing only on the training portion of each split, including scaling and text vocabulary, to [avoid leaking information](https://scikit-learn.org/stable/common_pitfalls.html#data-leakage).

Keep the simplest candidate that meets your accuracy and prediction-speed needs. Use validation results to choose settings, then evaluate the selected model on the reserved test set. The [datasets guide](./datasets-ml) explains those roles.
