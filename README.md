# Parameter-Isolation-for-CL
Project made for undergraduate thesis

## Table of Contents
1. [Introduction](#introduction)
2. [Catastrophic Forgetting](#cf)
3. [Continual Learning](#cl)
4. [Parameter Isolation](#pi)
5. [Implementing Parameter Isolation Methods](#implement)
6. [Datasets](#datasets)

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

<a name="datasets"></a>
## Datasets 
  
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
