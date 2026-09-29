---
title: Neural Network Tuning
short_title: Neural Network Tuning
description: Diagnose stalled training, deteriorating validation performance, and NaN or infinity errors in neural networks.
---

Start with what the model is doing. Record training and validation performance as learning proceeds, then use the symptom below to choose your next check. Loss measures the error the optimizer tries to reduce; also track a metric that reflects your task, such as classification accuracy or prediction error in the original units.

Keep the data split fixed while comparing settings. Use validation data to choose a model and reserve the test set for the final evaluation. The [datasets guide](./datasets-ml) explains how to separate those roles.

## Contents

| Symptom | Where to start |
|---|---|
| [Training stalls](#training-stalls) | Check the data and loss, then try fitting a small batch. |
| [Validation performance deteriorates](#validation-performance-deteriorates) | Check the comparison, then inspect whether training keeps improving as validation gets worse. |
| [Calculations produce invalid numbers](#calculations-produce-invalid-numbers) | Find the first NaN or infinity and the operation that produced it. |

## Training Stalls

A loss that barely moves can result from a data error, a broken parameter update, or settings that make learning slow. A plateau after substantial improvement can also mean the current model has learned as much as it can from the available inputs. Establish which situation you have before adding layers.

<a id="data-normalization"></a><a id="normalization"></a>
<a id="activation-function"></a><a id="activation"></a>
<a id="loss-function"></a><a id="loss"></a>

### Check the Inputs, Targets and Loss

Inspect a few examples after preprocessing. Confirm that each input still has the correct target and that the model's output shape matches what the loss expects. For a classifier, check the class encoding and whether the loss expects raw scores, called logits, or probabilities. Applying a probability transform twice changes the objective. The [Keras loss documentation](https://keras.io/api/losses/) shows how these conventions are specified.

For continuous predictions, a linear output with mean squared error is one starting point. An [autoencoder](./deep-autoencoder) instead uses its input as the reconstruction target, with an output activation and loss suited to those values.

For numerical inputs with very different scales, try standardization or another transformation suited to the data. Estimate its parameters, such as means and standard deviations, from the training set. Apply that fitted transformation unchanged to validation and test data. [Fitting preprocessing on held-out data leaks information](https://scikit-learn.org/stable/common_pitfalls.html#data-leakage) into model development.

### Try Fitting a Small, Fixed Batch

Take a small set of clean training examples, such as ten, and train repeatedly on that same set. Temporarily turn off augmentation and regularization that deliberately make fitting harder. For a model with enough capacity and compatible targets, the training loss should fall substantially.

If it barely changes, check that gradients reach the intended parameters and that an optimizer step changes their values. Inspect whether the labels match the inputs. A model that can fit the small batch has passed one diagnostic; it still needs to learn useful patterns across the full dataset. [Stanford's training notes](https://cs231n.github.io/neural-networks-3/#sanity) describe this check and its limits.

<a id="learning-rate"></a><a id="lrate"></a>
<a id="updater-and-optimization-algorithm"></a><a id="updater"></a>
<a id="minibatch-size"></a><a id="minibatch"></a>
<a id="policies-and-scheduling"></a>

### Inspect the Updates Before Changing the Architecture

Compare short runs with a smaller and a larger learning rate, starting from the same initial model and data order. Large swings or rising loss can indicate updates that are too large; extremely slow progress can indicate updates that are too small. Judge the curves alongside gradient and parameter changes, since several failures can produce the same curve.

A minibatch is the group of examples used for one update. Changing its size changes both the update's noise and the number of updates in an epoch, one pass through the training set. Record both epochs and update counts when comparing runs. For learning-rate schedules, check whether the framework counts epochs or optimizer steps; adding GPUs does not by itself determine a schedule adjustment.

<a id="weight-initialization"></a><a id="weight"></a>

Inspect activations and gradients across layers. Saturated activations or many inactive ReLU units can impede learning. Check input scale and initialization together; initialization methods such as He for ReLU and Glorot/Xavier for tanh are useful starting points. Choose and tune the optimizer as a whole, for example SGD with momentum or Adam, and keep its settings recorded with the run.

<a id="recurrent-neural-networks-truncated-backpropagation-through-time"></a><a id="rnn"></a>

For recurrent networks, truncated backpropagation limits how far gradients travel through a sequence. A shorter window reduces memory use but can prevent the model from learning dependencies across that boundary. See the [LSTM training explanation](./lstm) before changing the window.

## Validation Performance Deteriorates

If training improves while validation worsens, the model may be fitting details that fail to generalize. First establish that the measurements and data split support that interpretation.

### Make the Comparison Consistent

Evaluate checkpoints on a fixed validation set in the framework's evaluation mode. Dropout and batch normalization behave differently during training and evaluation. Training-time augmentation can also make training examples harder than validation examples.

For a direct comparison, evaluate the same saved checkpoint on both datasets under matching conditions, and calculate the same data loss or task metric. A logged training loss averaged over a changing model is a different measurement from validation loss calculated after the epoch. The [Keras FAQ](https://keras.io/getting_started/faq/#why-is-my-training-loss-much-higher-than-my-testing-loss) explains these differences.

Check whether the split represents the prediction task. A future-data evaluation should preserve time order; records from the same person may need to stay in one split. If the validation population differs from training, investigate that difference alongside overfitting.

<a id="number-of-epochs-and-number-of-iterations"></a><a id="epochs"></a>

### Keep the Best Validation Checkpoint

These illustrative losses are measured on fixed datasets with the same evaluation procedure:

| Epoch | Training loss | Validation loss |
|---|---:|---:|
| 1 | 0.80 | 0.90 |
| 5 | 0.40 | 0.50 |
| 10 | 0.20 | 0.65 |

The epoch-5 checkpoint has the lowest validation loss among the three. Continuing to epoch 10 improves training loss while worsening validation loss, a pattern consistent with overfitting.

Early stopping can stop training after the validation metric fails to improve for a chosen number of checks. Allow for noise, and explicitly save or restore the best checkpoint. In [Keras EarlyStopping](https://keras.io/api/callbacks/early_stopping/), `restore_best_weights=True` restores the best weights; the default is `False`.

<a id="regularization"></a>
<a id="restricted-boltzmann-machines-rbms"></a>
<a id="choosing-layer-width-and-depth"></a><a id="rbm"></a>

### Reduce Overfitting and Recheck

Try a smaller model, stronger regularization, or more representative training data. Regularization can include weight penalties or dropout; tune its strength because excessive regularization can also prevent useful learning. Data augmentation should preserve the target, as a crop that removes the object being classified can create a mislabeled example.

Compare layer widths and depths using validation performance. For autoencoders, compare bottleneck sizes against reconstruction quality on held-out examples. The [TensorFlow overfitting tutorial](https://www.tensorflow.org/tutorials/keras/overfit_and_underfit) demonstrates capacity and regularization experiments.

<a id="visiblehidden-unit"></a><a id="dbn"></a>

If you are using RBM pretraining, check the model's visible and hidden unit distributions and evaluate whether pretraining improves your result. The [restricted Boltzmann machine guide](./restricted-boltzmann-machine) explains that separate training procedure.

<a id="nan-not-a-number-errors-in-scoring"></a>
<a id="nan-and-infinity-errors"></a><a id="NaN"></a>

## Calculations Produce Invalid Numbers

NaN means "not a number." Infinity and NaN can spread through later calculations, so the final loss may only reveal a failure that happened earlier.

### Locate the First Nonfinite Value

Check that inputs and targets contain finite values after preprocessing. Then inspect the forward computation, followed by gradients and the parameter update. Stop at the first operation that changes finite values into NaN or infinity. [TensorFlow's numerical checks](https://www.tensorflow.org/api_docs/python/tf/debugging/enable_check_numerics) can report the operation that produces the invalid result.

### Match the Fix to the Operation

Invalid arithmetic such as `0/0` or the logarithm of a negative real number can produce NaN. Overflow can produce infinity, and later arithmetic such as infinity minus infinity can produce NaN. Underflow can round a tiny value to zero; it does not by itself produce NaN. These are distinct [floating-point conditions](https://numpy.org/doc/stable/reference/generated/numpy.seterr.html).

For a zero denominator, check why it became zero and what the formula should mean in that case. For unstable probability calculations, use a numerically stable loss implementation that accepts logits. If values first become extreme after a parameter update, inspect the learning rate and gradient magnitudes.

<a id="gradient-clipping"></a><a id="gradient-normalization"></a>

### Check Gradient Scale and Precision

Gradient clipping limits excessively large gradients before an update. Use it when large finite gradients are the problem; clipping cannot repair a NaN already produced by invalid arithmetic.

With float16 training, small gradients can underflow. Mixed-precision training commonly uses loss scaling to keep them representable. When clipping scaled gradients, unscale them first so the threshold applies to the original gradient magnitudes. [PyTorch's mixed-precision examples](https://docs.pytorch.org/docs/2.9/notes/amp_examples.html#gradient-clipping) show this order.

After fixing the cause, restart from a checkpoint whose model and optimizer state are finite. Check that the operation now stays finite, then confirm that training and validation performance improve under the corrected setup.
