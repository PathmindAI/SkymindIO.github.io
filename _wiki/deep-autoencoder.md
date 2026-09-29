---
title: Deep Autoencoders
short_title: Deep Autoencoders
description: How an encoder learns a compressed representation and a decoder reconstructs the input, with a worked example and runnable Python.
---

Give an autoencoder an image of a handwritten digit, and ask it to reproduce the pixel values. The encoder turns the image into a code. The decoder uses that code to reconstruct the image. Training adjusts both networks to reduce the difference between the reconstruction and the original.

This guide uses a code with fewer values than the input. That narrow representation is the bottleneck: it limits how much can pass directly from encoder to decoder. A deep autoencoder uses several hidden layers to build and decode the representation. The [neural-network guide](./neural-network#deep-learning) explains what those extra layers do.

```text
input x → encoder → compressed code h → decoder → reconstruction x_hat
   └──────────────── target for reconstruction loss ─────────────────┘
```

## Encoding Input Data

A 28-by-28-pixel [MNIST](./mnist) image contains 784 pixel values. Flatten those values into a vector and pass them through layers of weighted sums and activations. One possible encoder has these layer widths:

```text
784 (input) → 1000 → 500 → 250 → 100 → 30 (code)
```

Each number gives the count of values produced by that layer. The first hidden layer has **1,000 neurons**. In a fully connected layer, each neuron has a weight for each of its 784 inputs, plus one bias. Its parameter count is therefore:

```text
weights: 784 × 1,000 = 784,000
biases:        1,000 =   1,000
total:                  785,000 parameters
```

For a dense layer with `n` inputs and `m` outputs, the count is `n * m + m` when every output has a bias. The [Keras Dense documentation](https://keras.io/api/layers/core_layers/dense/) describes these weights and biases. The 785,000 parameters belong to this first layer; later layers add their own parameters.

These widths are design choices. The encoder can expand the representation in an early layer before narrowing it to the code. Its trained weights are shared across images; the activation values change with each image.

## The Compressed Representation

The encoder above produces 30 values for each image. This vector is also called a latent representation: values computed inside the model that carry information used by the decoder. Its coordinates are learned combinations of input features, so a coordinate need not correspond to a named property such as stroke thickness.

The training objective determines what the encoder has an incentive to retain. With pixel reconstruction error, preserving a background pattern can matter as much as preserving a stroke if they contribute similar amounts to the loss. A compact representation therefore needs evaluation for the task in which you plan to use it. [The Deep Learning textbook](https://www.deeplearningbook.org/contents/autoencoders.html) explains the bottleneck and why reconstruction alone can still produce unhelpful features.

Compare different code widths on validation data. A smaller code may lose useful detail; a larger one may make reconstruction easier while preserving more information than you need. Here, compression means fewer coordinates. Actual storage savings also depend on how many bits encode each value and how the decoder is stored or shared.

## Decoding Representations

The decoder maps the 30-value code back to 784 predicted pixel values. A decoder that mirrors the encoder's widths would be:

```text
30 (code) → 100 → 250 → 500 → 1000 → 784 (reconstruction)
```

Mirroring is one architecture choice. The decoder's output shape must match the input being reconstructed. For this image, reshape the 784 predictions into a 28-by-28 grid and compare it with the original. The decoder learns an approximate reconstruction from the code; details discarded by the encoder may be lost.

## Training on Reconstruction Error

For each ordinary reconstruction-training example, the target is the input itself. If `x` is the input, the encoder computes `h = f(x)` and the decoder computes `x_hat = g(h)`. A mean squared error over `d` input values is:

```text
loss = sum((x_hat[j] - x[j]) ** 2 for j in range(d)) / d
```

Backpropagation carries the loss gradient through the decoder and into the encoder. The optimizer uses those gradients to update both networks. The original pixel values supply the training targets without digit-class labels. [TensorFlow's autoencoder tutorial](https://www.tensorflow.org/tutorials/generative/autoencoder) demonstrates this joint training.

### A Small Numerical Example

A `2 → 1 → 2` autoencoder makes the arithmetic small enough to follow. This is a shallow example of the same encoding and decoding operations used in a deeper network. Choose the starting weights by hand and feed it `x = [1, 2]`.

The encoder has weights `[0.5, 0.25]`, a zero bias, and a ReLU activation, which returns `max(0, value)`:

```text
h = max(0, 0.5 × 1 + 0.25 × 2 + 0) = 1
```

The decoder has weights `[0.8, 1.5]` and two zero biases. Its outputs are linear because this example reconstructs continuous values:

```text
reconstruction = [0.8 × 1 + 0, 1.5 × 1 + 0] = [0.8, 1.5]
target         = [1, 2]
mean squared error = ((0.8 − 1)² + (1.5 − 2)²) / 2 = 0.145
```

The encoder has `2 × 1 + 1 = 3` parameters. The decoder has `1 × 2 + 2 = 4`, making **7 parameters** in total. The hidden value `h = 1` is an activation computed for this input; the weights and biases are the quantities training adjusts.

### Runnable Python

This Python 3 example needs no additional packages. It reproduces the calculation and takes one gradient-descent step on the same input. For a mean squared error over two outputs, the derivative with respect to each prediction equals that prediction minus its target. Backpropagation then uses the decoder weights to calculate the derivative with respect to the hidden value.

```python
x = [1.0, 2.0]
encoder_weights = [0.5, 0.25]
encoder_bias = 0.0
decoder_weights = [0.8, 1.5]
decoder_biases = [0.0, 0.0]
learning_rate = 0.01


def forward():
    z = sum(w * value for w, value in zip(encoder_weights, x)) + encoder_bias
    h = max(0.0, z)
    prediction = [w * h + b for w, b in zip(decoder_weights, decoder_biases)]
    loss = sum((pred - target) ** 2 for pred, target in zip(prediction, x)) / len(x)
    return z, h, prediction, loss


z, h, prediction, loss = forward()
print("First dense layer parameters:", 784 * 1000 + 1000)
print("Toy model parameters:", len(encoder_weights) + 1
      + len(decoder_weights) + len(decoder_biases))
print("Hidden value:", h)
print("Before:", prediction, "MSE:", round(loss, 6))

# Compute every gradient using the original weights, before updating them.
prediction_grad = [2 * (pred - target) / len(x)
                   for pred, target in zip(prediction, x)]
decoder_weight_grad = [grad * h for grad in prediction_grad]
hidden_grad = sum(grad * w for grad, w in zip(prediction_grad, decoder_weights))
z_grad = hidden_grad if z > 0 else 0.0
encoder_weight_grad = [z_grad * value for value in x]

encoder_weights = [w - learning_rate * grad
                   for w, grad in zip(encoder_weights, encoder_weight_grad)]
encoder_bias -= learning_rate * z_grad
decoder_weights = [w - learning_rate * grad
                   for w, grad in zip(decoder_weights, decoder_weight_grad)]
decoder_biases = [b - learning_rate * grad
                  for b, grad in zip(decoder_biases, prediction_grad)]

_, _, prediction, loss = forward()
print("After:", [round(value, 6) for value in prediction], "MSE:", round(loss, 6))
```

Output:

```text
First dense layer parameters: 785000
Toy model parameters: 7
Hidden value: 1.0
Before: [0.8, 1.5] MSE: 0.145
After: [0.847789, 1.592173] MSE: 0.094745
```

The update reduces reconstruction error for this example. To learn a useful representation, train on many examples, then measure reconstruction error on held-out data to see how the model handles new inputs.

### Training Nuances

Choose the output activation and loss to match the values being reconstructed. A linear output can predict unrestricted continuous values with mean squared error. For pixels scaled to the interval from 0 to 1, a sigmoid output constrains predictions to that interval. The reconstruction loss then measures the discrepancy you want training to reduce.

Use training data to adjust the weights and validation data to choose the code width and other settings. Inspect held-out reconstructions alongside the numerical loss: an acceptable average can hide badly reconstructed examples. Reserve the test set for the final evaluation, as described in the [datasets guide](./datasets-ml). The [tuning guide](./neural-network-tuning) helps diagnose stalled training or worsening validation performance.

<a id="use-cases"></a>

## Using the Learned Representation

### Image Search

You can encode images, store their codes, and retrieve images whose codes are nearby under a chosen distance measure. Evaluate the retrieved images against what readers mean by "similar." Reconstruction training rewards recovery of pixel values, so nearby codes may reflect background or texture when the search task needs object identity.

### Data Compression

A code with 30 values has fewer coordinates than the original 784-value image. To assess it as a storage format, measure the encoded bit count and reconstruction quality together, including any model-storage cost that the application must bear. Changing code width or numerical precision changes that tradeoff.

<a id="topic-modeling--information-retrieval-ir"></a>

### Document Retrieval

An encoder can also turn a document's word-count vector into a shorter code and train a decoder to reconstruct that representation. Test whether nearby codes retrieve relevant documents. A reconstruction objective does not by itself assign interpretable topics or establish that a retrieved document answers a question.

## Historical Note: RBM Pretraining

In their [2006 paper, *Reducing the Dimensionality of Data with Neural Networks*](https://www.cs.toronto.edu/~hinton/absps/science.pdf), Geoffrey Hinton and Ruslan Salakhutdinov trained a stack of restricted Boltzmann machines layer by layer. Those weights initialized a deep encoder and a mirrored decoder, which they then fine-tuned together with backpropagation to reduce reconstruction error.

That pretraining procedure helped initialize deep networks. It is one way to prepare their weights; the encoder and decoder can also be trained directly on reconstruction loss, as in the example above. The [restricted Boltzmann machine guide](./restricted-boltzmann-machine) explains the separate RBM training method.
