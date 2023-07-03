# Parameter-Isolation-for-CL

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
