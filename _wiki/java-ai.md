---
title: Java Tools for Deep Learning, Machine Learning and AI
short_title: Java Tooling for AI
description: A guide to using the Java Virtual Machine for artificial intelligence.
---

Why should you use JVM languagues like Java, [Scala](scala-ai), Clojure or Kotlin to build AI and machine-learning solutions?

Java is the [most widely used programming language in the world](https://www.tiobe.com/tiobe-index/). Large organizations in the public and private sector have enormous Java code bases, and rely heavily on the JVM as a compute environment. In particular, much of the open-source big data stack is written for the JVM. This includes [Apache Hadoop](http://hadoop.apache.org/) for distributed data management; [Apache Spark](apache-spark-deep-learning) as a distributed run-time for fast ETL; [Apache Kafka](https://kafka.apache.org/) as a message queue; [ElasticSearch](https://www.elastic.co/), [Apache Lucene](https://lucene.apache.org/) and [Apache Solr](http://lucene.apache.org/solr/) for search; and [Apache Cassandra](http://cassandra.apache.org/) for data storage to name a few. The tools below give you powerful ways to leverage machine learning on the JVM.

{% include wiki-inline-cta.html %}

## Deep Learning & Neural Networks

Deep learning usually refers to deep artificial neural networks. [Neural networks](neural-network) are a type of machine learning algorithm loosely modeled on the neurons in the human brain. 

### TensorFlow-Java

[TensorFlow provides a Java API](https://www.tensorflow.org/install/lang_java). While it is not as fully developed as TensorFlow's Python API, progress is being made. Karl Lessard is leading efforts to adapt TensorFlow to the JVM. Those interested can join the TensorFlow Java SIG or this [Tensorflow JVM Gitter channel](https://gitter.im/tensorflow/sig-jvm). [TensorFlow Serving](https://www.tensorflow.org/tfx/guide/serving) is a flexible, high-performance serving system for machine learning models, designed for production environments. [TensorFlow-Java's Github repository can be found here](https://github.com/tensorflow/java/) and [here](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/java). Companies such as Facebook are active on the TensorFlow SIG led by [Karl Lessard](https://github.com/karllessard).

### Neuroph

[Neuroph](http://neuroph.sourceforge.net/) is an open-source Java framework for neural networks. Developers can create neural nets with the Neuroph GUI. The Neuroph API documentation also explains how neural networks work.

### MXNet

[Apache MXNet](https://mxnet.apache.org/api) has a [Java API](https://cwiki.apache.org/confluence/display/MXNET/MXNet+Java+Inference+API) as well as many other bindings. It is backed by Carnegie Mellon and Amazon as well as the Apache Foundation. 

### Deep Java Library (DJL)

[Deep Java Library](https://djl.ai) is another Java-focused deep learning dev tool [introduced by Amazon](https://towardsdatascience.com/introducing-deep-java-library-djl-9de98de8c6ca). 

### Deeplearning4j

Eclipse [Deeplearning4j](https://github.com/deeplearning4j) is a DSL that allows users to configure neural networks in Java. It was created by the US-based startup Skymind. 

### Computer Vision JSR

Frank Greco of IBM and Zoran Severac are leading an effort to define a [computer vision API for Java](https://jcp.org/en/jsr/detail?id=381).

## Machine Learning Model Servers

### Seldon

Seldon is a Java-focused, open-source, [machine learning model server](https://www.seldon.io/) that integrates with Kubernetes. [Seldon's Github repository](https://github.com/SeldonIO). Its name references the godfather of psycho-historians, Hari Seldon, of Isaac Asimov's Foundation series, who uses math to predict the future. 

### Kubeflow

[Kubeflow](https://github.com/kubeflow) is an open, community-driven project to make it easy to deploy and manage an ML stack on Kubernetes. [Kubeflow pipelines](https://github.com/kubeflow/pipelines) are reusable end-to-end ML workflows (including models and data transforms) built using the Kubeflow Pipelines SDK.

### Amazon Sagemaker

[Amazon Sagemaker](https://docs.aws.amazon.com/sagemaker/latest/dg/how-it-works-hosting.html) is a tool for building, training and deploying machine learning models to production. 

### MLeap

[MLeap](http://mleap-docs.combust.ml/) is an open-source project that helps deploy Spark pipelines, including ML models, to production. 

## Expert Systems

An expert system is also called a rules-based system. The rules are typically if-then statements; i.e. if this condition is met, then perform this action. An expert system usually comprises hundreds or thousands of nested if-then statements. Expert systems were a popular form of AI in the 1980s. They are good at modeling static and deterministic relationships; e.g. the tax code. However, they are also brittle and they require manual modification, which can be slow and expensive. Unlike, machine-learning algorithms, they do not adapt as they are exposed to more data. They can be a useful complement to a machine-learning algorithm, codifying the things that should always happen a certain way.

### Drools

[Drools](https://www.drools.org/) is a business rules management system backed by Red Hat.

## Solvers

### OptaPlanner

[Optaplanner](https://www.optaplanner.org/) is an AI constraint solver written in Java. It is a lightweight, embeddable planning engine that includes algorithms such as Tabu Search, Simulated Annealing, Late Acceptance and other metaheuristics with very efficient score calculation and other state-of-the-art constraint solving techniques.

## Natural-Language Processing

Natural language processing (NLP) refers to applications that use computer science, AI and computational linguistics to enable interactions between computers and human languages, both spoken and written. It involves programming computers to process large natural language corpora (sets of documents).

Challenges in natural language processing frequently involve natural language understanding (NLU) and natural language generation (NLG), as well as connecting language, machine perception and dialog systems.

### OpenNLP

[Apache OpenNLP](https://opennlp.apache.org/) is a machine-learning toolkit for processing natural language; i.e. text. The official website provides API documentation with information on how to use the library.

### Stanford CoreNLP

[Stanford CoreNLP](https://stanfordnlp.github.io/CoreNLP/) is the most popular Java natural-language processing framework. It provides various tools for NLP tasks. The official website provides tutorials and documentation with information on how to use this framework.

## Machine Learning

Machine learning encompasses a wide range of algorithms that are able to adapt themselves when exposed to data, this includes random forests, gradient boosted machines, support-vector machines and others.

### SMILE

[SMILE](https://github.com/haifengl/smile) stands for Statistical and Machine Intelligence Learning Engine. SMILE was create by Haifeng Lee, and provides fast, scalable machine learning for Java. SMILE uses ND4J to perform scientific computing for large-scale tensor manipulations. It includes algorithms such as support vector machines (SVMs), [decision trees](decision-tree), [random forests](random-forest) and gradient boosting, among others.

### Weka 

[Weka](http://www.cs.waikato.ac.nz/ml/weka/) is a collection of machine learning algorithms that can be applied directly to a dataset, through the Weka GUI or API. The WEKA community is large, providing various tutorials for Weka and machine learning itself. 

### MOA (Massive On-line Analysis)
[MOA (Massive On-line Analysis)](https://moa.cms.waikato.ac.nz/) is for mining data streams.
