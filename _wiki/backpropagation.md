---
title: A Beginner's Guide to Backpropagation in Neural Networks
short_title: Backpropagation
description: A beginner's reference to Backpropagation, a key algorithm in training neural networks.
---

*A man of genius makes no mistakes. His errors are volitional and are the portals to discovery. - James Joyce* 

Backpropagation is the central mechanism by which [artificial neural networks](./neural-network) learn. It is the messenger telling the neural network whether or not it made a mistake when it made a prediction. 

*To propagate* is to transmit something (light, sound, motion or information) in a particular direction or through a particular medium. *To backpropagate* is to to transmit something in response, to send information back upstream -- in this case, with the purpose of correcting an error. When we discuss backpropagation in deep learning, we are talking about the transmission of information, and that information relates to the error produced by the neural network when it makes a guess about data. Backpropagation is synonymous with correction. 

Untrained neural network models are like new-born babies: They are created ignorant of the world, and it is only through exposure to the world, experiencing it, that their ignorance is slowly relieved. Algorithms experience the world through data. So by training a neural network on a relevant dataset, we seek to decrease its ignorance. The way we measure progress is by monitoring the error produced by the network each time it makes a prediction. The way we achieve progress is by minimizing that error gradually in small steps. The errors are portals to discovery.

The knowledge of a neural network with regard to the world is captured by its weights, the parameters that alter input data as its signal flows through the neural network towards the net's final layer, which will make a decision about that input. Those decisions are often wrong, because the parameters transforming the signal into a decision are poorly calibrated; they haven't learned enough yet. Forward propagation is when a data instance sends its signal through a network's parameters toward the prediction at the end. Once that prediction is made, its distance from the ground truth (error) can be measured.

So the parameters of the neural network have a relationship with the error the net produces, and when the parameters change, the error does, too. We change the parameters using optimization algorithms. A very popular optimization method is called [gradient descent](https://en.wikipedia.org/wiki/Gradient_descent){:target="_blank"}, which is useful for finding the minimum of a function. We are seeking to minimize the error, which is also known as the *loss function* or the *objective function*.

<a class="w-button button-skilcta" href="https://pathmind.com" style="width:75%; margin-top: 15px;" target="_blank">Learn to build AI in Simulations >></a>

## Backpropagation

A neural network propagates the signal of the input data forward through its parameters towards the moment of decision, and then *backpropagates* information about the error, in reverse through the network, so that it can alter the parameters. This happens step by step: 

* The network makes a guess about data, using its parameters
* The network's is measured with a loss function
* The error is backpropagated to adjust the wrong-headed parameters

You could compare a neural network to a large piece of artillery that is attempting to strike a distant object with a shell. When the neural network makes a guess about an instance of data, it fires, a cloud of dust rises on the horizon, and the gunner tries to make out where the shell struck, and how far it was from the target. That distance from the target is the measure of error. The measure of error is then applied to the angle and direction of the gun (parameters), before it takes another shot. Operations researchers will recognize that backpropagation resembles ideas in control theory regarding feedback and optimization.

Backpropagation takes the error associated with a wrong guess by a neural network, and uses that error to adjust the neural network's parameters in the direction of less error. How does it know the direction of less error?

## Gradient Descent

A *gradient* is a slope whose angle we can measure. Like all slopes, it can be expressed as a relationship between two variables: "y over x", or *rise over run*. In this case, the `y` is the error produced by the neural network, and `x` is the parameter of the neural network. The parameter has a relationship to the error, and by changing the parameter, we can increase or decrease the error. So the gradient tells us the change we can expect in `y` with regard to `x`. 

To obtain this information, we must use differential calculus, which enables us to measure *instantaneous rates of change*, in this case is the tangent of a changing slope expressing the relationship of the weight to the neural network's error. As the parameter changes, the error changes, and we want to move both variables in the direction of less error.

Obviously, a neural network has many parameters, so what we're really measuring are the *[partial derivatives](https://en.wikipedia.org/wiki/Partial_derivative){:target="_blank"}* of each parameter's contribution to the total change in error. 

What's more, neural networks have parameters that process the input data sequentially, one after another. Therefore, backpropagation establishes the relationship between the neural network's error and the parameters of the net's last layer; then it establishes the relationship between the parameters of the neural net's last layer those the parameters of the second-to-last layer, and so forth, in an application of the *[chain rule of calculus](https://en.wikipedia.org/wiki/Chain_rule){:target="_blank"}*. 

It's interesting to note that backpropagation in artificial neural networks has echoes in the functioning of biological neurons, which respond to rewards such as dopamine to reinforce how they fire; i.e. how they interpret the world. Dopaminergenic behavior tends to strengthen the ties between the neurons involved, and helps those ties last longer. While biological neurons do not perform backpropagation precisely (see further reading below), they appear to achieve the same effect by other means. That is, backpropagation allows artificial neural networks to roughly mimic certain narrow mechanisms of intelligence that are found in the example of intelligence that we have: the human brain. 

### Further Reading About Backpropagation

* [Backpropagation](https://brilliant.org/wiki/backpropagation/){:target="_blank"}
* [3BLUE1BROWN VIDEO: What is backpropagation really doing?](https://www.youtube.com/watch?v=Ilg3gGewQ5U)
* [Dendritic cortical microcircuits approximate the backpropagation algorithm](https://arxiv.org/abs/1810.11393v1)
* [Predictive Coding Can Do Exact Backpropagation on Any Neural Network](https://arxiv.org/abs/2103.04689)

## Other Posts on the Pathmind Wiki

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
* [Deep Reinforcement Learning](./deep-reinforcement-learning)
* [Generative Adversarial Networks (GANs)](./generative-adversarial-network-gan)
* [AI vs Machine Learning vs Deep Learning](./ai-vs-machine-learning-vs-deep-learning)
* [Multilayer Perceptrons (MLPs)](./multilayer-perceptron)
