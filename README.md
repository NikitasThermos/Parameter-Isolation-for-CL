# Parameter-Isolation-for-CL
Project made for undergraduate thesis

## Table of Contents
1. [Introduction](#introduction)
2. [Catastrophic Forgetting](#cf)
3. [Continual Learning](#cl)

<a name="introduction"></a>
## Introduction
The thesis focuses on one of Deep Learning's fields named Continual Learning. Continual Learning tries to alleviate a problem thant simple and traditional Deep Learning models experience named Catastrophic Forgetting. In the thesis, we try to define and understand the problem of Catastrophic Forgetting, find under which occasions it appears and the limitations it creates for the Deep Learning models. Then we present ways that we can avoid Catastrophic Forgetting with the use of Continual Learning. Further, this thesis focuses on one field of Continual Learning called Parameter Isolation, as we present how we can achieve Continual Learnign with Parameter Isolation methods, we implement two of the most popular Parameter Isolation methods and we make experiments with some of the most famous Continual Learning scenarios. 

<a name="cf"></a>
## Catastrophic Forgetting
Catastrophic Forgetting appearead from the very early days of Neural Networks when researches tried to train models on multiple tasks sequentially. They found out that the model's perfomance on the first task decreases rapidly while training for a second different task and in many cases the perfomance is getting worse than a random initialized model without any training. Catastrophic Forgetting can also appear in cases when you train a model for the same task but you input different datasets sequentially in which the instances from the different datasets have different distributions. In those cases the model may forget the instances of the first dataset, perfoming well only when it sees instances that are closes to the second datasset. 

<a name="cl"></a>
## Continual Learning
The filed of Continual Learning tried to develop different methods that we allow Deep Learning models to learn multiple tasks in sequence while preserving the perfomance on all tasks. Continual Learning methods can be classified in three categories:
* Replay Methods
* Regularization-based Methods
* Parameter Isolation Methods




## Methods

We have implemented two Parameter Isolation methods: 
* Progressive Neural Networks (PNNS)
* PackNet
  
and two baselines: 
* Finetune Training
* Indivindual Networks

All of the experiments were executed in Google Colab, and the notebooks can be found in the 'Notebooks' folder. We also provide implementations of the methods in Python module format, but they have not been extensively tested. To run any method you can execute the corresponding notebook or run the main python module for each method ('pnn.py', 'packnet.py', 'finetune.py', 'indivindualNets.py'). Note that if you run the python modules you need to specify the dataset you want to use. More information about the datasets can be found on the next section.

## Datasets

We used the following datasets for our experiments: 
* MNIST
* CIFAR100
* Flowers102

And we made four different experiment scenarios:
* Permuted MNIST (5 tasks / 10 classes per task)
* Split CIFAR100 (5 tasks / 20 classes per task)
* Split Flowers102 (6 tasks / 17 classes per task)
* Different Datasets (3 tasks, each one consists of the whole MNIST/CIFAR100/Flowers102 datasets)

If your run the modules you need to specify the dataset you want to use with the '--dataset' argument. Available options are 'pmnist', 'cifar100', 'flowers102', 'diffDatasets'
