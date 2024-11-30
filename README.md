# Parameter-Isolation-for-CL
Project made for undergraduate thesis

## Table of Contents
1. [Introduction](#introduction)
2. [Catastrophic Forgetting](#cf)
3. [Continual Learning](#cl)
4. [Parameter Isolation](#pi)
5. [Implementing Parameter Isolation Methods](#implement)
6. [Continual Learning Scenarios](#scenarios)
7. [Experiments](#experiments)

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

Replay methods try to keep high perfomance on all tasks by feeding the model instances (or pseudo-instances) from previously trained tasks while the model learns a new task. On the other hand, Regularization-based methods add regularization terms to the loss function that help avoiding uncontrollable updates of the model's parameters. Parameter isolation methods propose the use of different parameters of the model for each different task thus each task uses a different subset of the parameters. 

<a name="pi"></a>
## Parameter Isolation
The isolation of model's parameters eliminates any overlap between the tasks by giving an entirely different subset of the whole network to each task. There are some methods that allow some overlap between the tasks so that parts of the model that had trained on a previous task can help on the training of a feature task by transfer learning. In most occasions the Parameter Isolation methods can be split in two categories:
* Dynamic Architecture
* Fixed Network
  
In dynamic architectures new parameters can be added to the model when a new task arrives for training. These new parameters will be trained on the new task while the other parameters remain fixed to keep the perfomance on the previous tasks stable. In fixed networks, the initial network does not change and the isolation happens by just using a subset of the model's parameters for each task. 

<a name="implement"></a>
## Implementing Parameter Isolation Methods
We have implemented one method for each of the two main families of Parameter Isolation methods : 
* Progressive Neural Networks (Dynamic Architecture)
* PackNet (Fixed Network)

Progressive Neural Networks (PNNs) add a new subnetwork each time a new task is trained. When a subnetwork is trained on one task its parameters remain stable to keep the perfomance for that task. Each new subnetwork that it is added to the model is connected with the last trained subnetwork with lateral connetioncs. For example, while training on a second task a new subnetwork is created and connected with the subnetwork that was trained on the first task  One layer of the new subnetwork is connected with the lower level layer of the previous subnetwork. In the example, the second layer of the new subnetwork is connected to the first layer of the subnetwork trained on the first task. While the first subnetwork's parameters remain the same the connections between tbe networks are trained during the second task training. These connections decide what knowledge is usefull from the previous network and can help with the new task. While predicting for the second task the instance go through both the first network (to activate the lateral connections) and the second subnetwork but the output of the first subnetwork is ignored.

PackNet uses a single unchanble network. It trains the first task on the whole network and then it decides which are the top k most usefull parameters for that task and keeps only those for that task. Because it removes the biggest part of the network and only keeps a small amount of parameters for each task the parameters that have been selected are going through a second phase training, this time without all the other parameters. After that, the parameters selected for a task remain stable. This can happen until the remaining parameters are not enough to achieve a good perfomance on a new task.

<a name="scenarios"></a>
## Continual Learning Scenarios 

We used the following datasets for our experiments: 
* MNIST
* CIFAR100
* Flowers102

With these three datasets we created the following Continual Learning scenarios:

* Permuted MNIST (5 tasks / 10 classes per task)
* Split CIFAR100 (5 tasks / 20 classes per task)
* Split Flowers102 (6 tasks / 17 classes per task)
* Different Datasets (3 tasks, each one consists of the whole MNIST/CIFAR100/Flowers102 datasets)

The Permuted MNIST was created by adding a different permutation to the whole MNIST dataset five times, thus creating 5 tasks each containing the whole MNIST dataset but with a different permutation. For compliance with the other experiments and the architecture of the models we increased the size of the MNIST images to 32x32 and increased the channels to 3. The models are trained sequentially to each task and must keep good perfomance for each of the permuations.
CIFAR100 contains 100 different classes. We created 5 tasks with each containing all the instances of 20 classes and each class was included only in one task. Thus each task now contains a different subset of the classes. The models are trained sequentially and must not lose perfomance on the previous learned classes when new ones arrive.
In the same way we created the split Flowers102 experiments, with each task containing a different subset of the classes. But this time each class containts very few instances and the classes are imbalanced. 
For the last experiment we used the three datasets for each task. With that the first task is to learn the whole MNIST dataset, the second task is to learn the whole CIFAR100 dataset and for the last task is to learn the whole Flowers102 dataset

<a name="experiments"></a>
## Experiments
To find out if the two Parameter Isolation methods are perfoming well, we also created two baselines: 

* Finetune Training
* Indivindual Networks

In Finetune Trainig, we train sequentially one Neural Network on all tasks without using any Continual Learning techniques. The goal here is to see if Catastrophic Forgetting actually occurs in these experiments without the use of Continual Learning.
On the other hand, in the Indivindual Networks baseline we train a different Neural Network on each task. Here we can see how well an individual network can perform on each task and find out if indivindual networks can outperform the Continual Learning methods that try to fit all the tasks in one model.

All of the experiments were executed in Google Colab, and the notebooks can be found in the 'Notebooks' folder. We also provide implementations of the methods in Python module format.

<a name="results"></a>
## Results

* Permuted MNIST

| Method | Task 1 | Task 2 | Task 3 | Task 4 | Task 5 | AAC | Parameters |
| --- | --- | --- | --- | --- | --- | --- | --- |
| PNN | 96.54 | 96.50 | 96.88 | 96.64 | 96.46 | 96.60 | 103,203,030
| PackNet | 95.49 | 96.25 | 96.41 | 96.35 | 95.94 | 96.08 | 25,659,482
| Finetune | 17.15 | 19.63 | 29.76 | 58.39 | 96.27 | 44.24 | 23,528,522
| Indivindual Networks | 96.80 | 96.65 | 96.83 | 96.32 | 96.10 | 96.54 | 117,642,610 

* CIFAR100

| Method | Task 1 | Task 2 | Task 3 | Task 4 | Task 5 | AAC | Parameters |
| --- | --- | --- | --- | --- | --- | --- | --- |
| PNN | 58.80 | 63.28 | 62.61 | 59.88 | 63.51 | 61.61 | 103,279,980
| PackNet | 48.46 | 58.48 | 59.10 | 58.28 | 60.18 | 56.90 | 25,761,932
| Finetune | 1.80 | 3.27 | 4.85 | 7.05 | 65.55 | 16.50 | 23,549,012
| Indivindual Networks | 60.10 | 60.98 | 60.20 | 60.76 | 60.90 | 60.58 | 117,745,060

* Flowers102

| Method | Task 1 | Task 2 | Task 3 | Task 4 | Task 5 | Task 6 | AAC | Parameters |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | 
| PNN | 49.52 | 41.27 | 49.05 | 75.31 | 43.59 | 40.35 | 49.84 | 138,107,493
| PackNet | 55.68 | 32.36 | 35.74 | 70.20 | 39.70 | 35.47 | 44.85 | 25.766.030
| Finetune | 6.36 | 8.31 | 5.76 | 3.31 | 6.97 | 33.14 | 10.64 | 23,717,030
| Indivindual Networks | 53.85 | 38.82 | 38,27 | 60.42 | 36.81 | 32.84 | 43.50 | 141,257,190

* Different Datasets

| Method | MNIST | CIFAR100 | Flowers102 | AAC | Parameters |
| --- | --- | --- | --- | --- | --- | 
| PNN | 98.85 | 58.76 | 48.40 | 68.70 | 47,967,300
| PackNet | 98.89 | 55.44 | 42.48 | 65.60 | 25,991,998
| Finetune | 1.2 | 9.59 | 37.80 | 16.19 | 23,717,030 
| Indivindual Networks | 98.95 | 60.20 | 35.40 | 64.85 | 70,958,484 




