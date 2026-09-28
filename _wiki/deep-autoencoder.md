---
title: Deep Autoencoders
short_title: Deep Autoencoders
description: How autoencoders encode and reconstruct inputs, with a worked parameter count and runnable Python example.
---

An autoencoder learns to reconstruct its input. Its encoder turns the input into a representation, and its decoder uses that representation to predict the original values. A deep autoencoder passes those values through several hidden layers. The [neural-network guide](./neural-network#deep-learning) explains how successive layers transform a representation.

Early deep autoencoders used [restricted Boltzmann machines](./restricted-boltzmann-machine) to initialize their weights. An encoder and decoder can also be trained together by backpropagation, using a loss that measures reconstruction error. [TensorFlow's introductory tutorial](https://www.tensorflow.org/tutorials/generative/autoencoder) demonstrates this approach.

### Encoding Input Data

Consider an encoder for a 28-by-28-pixel image from [MNIST](./mnist). Flattening the image produces 784 input values. One possible sequence of layer widths is:

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

The encoder above first expands the representation, then reduces it to 30 values. Those widths are design choices. Compression comes from the narrow code, often called the bottleneck; an expanded first layer is optional. Compare candidate widths using reconstruction error on validation data.

### Decoding Representations

The decoder maps the 30-value code back to 784 predicted pixel values. A decoder that mirrors the encoder's widths would be:

```text
30 (code) → 100 → 250 → 500 → 1000 → 784 (reconstruction)
```

Its output has the same number of values as the input image. The training target is the original image. Backpropagation carries the reconstruction-loss gradient through the decoder and into the encoder, adjusting both sets of weights.

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

Choose a reconstruction loss that fits the data and output activation. Mean squared error measures differences between continuous values, as in the example above. Select the learning rate and layer widths using validation data, and reserve the test set for the final evaluation. The [datasets guide](./datasets-ml) explains those roles.

## Use Cases

### Image Search

The encoder above represents each image with 30 numbers; a different architecture can use a different code width.

Image search, therefore, becomes a matter of uploading an image, which the search engine will then compress to 30 numbers, and compare that vector to all the others in its index. 

Vectors containing similar numbers will be returned for the search query, and translated into their matching image. 

### Data Compression

A more general case of image compression is data compression. Deep autoencoders are useful for [semantic hashing](https://www.cs.utoronto.ca/~rsalakhu/papers/semantic_final.pdf){:target="_blank"}, as discussed in this paper by Geoff Hinton.

### Topic Modeling & Information Retrieval (IR)

Deep autoencoders are useful in topic modeling, or statistically modeling abstract topics that are distributed across a collection of documents. 

This, in turn, is an important step in question-answer systems like Watson.

A document can be represented by word counts and encoded into a shorter vector. The decoder learns to reconstruct the document representation, using a loss suited to that representation. Scaling counts into a range between 0 and 1 does not by itself make them probabilities.

The encoder architecture determines the number of values in the code. Historical work on semantic hashing used RBM pretraining; the [paper linked above](https://www.cs.utoronto.ca/~rsalakhu/papers/semantic_final.pdf) describes that particular method.

Each document’s number set, or vector, is then introduced to the same vector space, and its distance from every other document-vector measured. Roughly speaking, nearby document-vectors fall under the same topic. 

For example, one document could be the “question” and others could be the “answers,” a match the software would make using vector-space measurements.
