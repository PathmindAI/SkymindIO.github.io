---
title: A Beginner's Guide to Supervised Learning
short_title: Supervised Learning
description: Beginner's reference to machine learning when labels are known - supervised learning.
---

You may have heard the term "supervised learning". It is a way to make an algorithm learn something, but it's not the algorithm, exactly.

In a supervised learning scenario, you have input data that has been labeled, like you might scribble a name on the back of a photograph so that future generations can identify your great aunt.

Imagine your algorithm is taking a test. The labels are the correct answers to the test (the name of the person), and the input data (the photographs) is the set of questions.

Your algorithm is going to take a guess each time it sees a data instance (a photograph), and that guess will be checked against the correct answer, the label, or name. The act of supervision, then, is this checking of guesses against correct answers. When the algorithm makes an incorrect guess, that can be detected, and the algorithm itself can be adjusted to make it guess better. This is the *learning* part of *machine learning*.  

(We don't always have a labeled dataset, nor do we know the answers to our questions, so not all learning can be supervised. [Unsupervised learning](./unsupervised-learning) is a topic for another post.)

The input data is what you want to make predictions about, and the labels are those precise predictions, attached to each relevant instance of the input.

Here are some examples of input-label pairs and what we would call this use case:

* Photo - Name of object in photo (e.g. face - name) - object recognition
* Transaction - "fraud" or "not_fraud" - fraud detection
* Text - "angry" or "content" - sentiment analysis
* Sound file - name of person speaking - voice recognition

## What you need for supervised learning

In order to perform supervised learning, you need a labeled dataset, and known answers to the questions you are asking.

These conditions are not always met. We don't always possess a labeled dataset for the predictions we seek to make. Worse than that, we don't always *know* the answers. For example, fraud and cybersecurity are two rapidly evolving fields, where attackers consistently search for new exploits. The institutions targeted by those new attacks could not possibly have a dataset in which they are labeled as attacks, because they have never been seen before. But they are attacks nevertheless.

The simpler problem to solve is that of labeling data where the answers are known. Sometimes these data labeling tasks require expertise (e.g. please circle the pixels in the X-ray that may indicate a cancerous tumor); sometimes they can be done by just about anyone (e.g. think Captcha).
