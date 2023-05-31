
from typing import List
import argparse
from Datasets.datasets import *

from tqdm import tqdm

import torch
import torch.nn as nn

import torchvision
import torchvision.models as models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', choices = ['pmnist', 'cifar100', 'flowers102', 'diffDatasets'], required = True, help = 'The dataset to be used for the experiment')

class IndivindualNets:
  def __init__(self, num_models, num_outputs, epochs):
    self.num_models = num_models

    if isinstance(num_outputs, List):
      self.num_outputs = num_outputs
    else:
      self.num_outputs = [num_outputs for _ in range(self.num_models)]

    self.models = [models.resnet34(weights = None) for _ in range (self.num_models)]

    #Replace the classfier for each model to match the number of classes of each task
    for index, model in enumerate(self.models):
      model.fc = nn.Linear(model.fc.in_features, self.num_outputs[index])
    
    self.epochs = epochs
    self.lossFunc = nn.CrossEntropyLoss()
    self.task_idx = 0
  
  def train(self, trainloader, testloader, lr):
    self.task_idx += 1
    model = self.models[self.task_idx - 1]
    model.train()
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr = lr)

    for epoch in range(self.epochs):
      for inputs, labels in tqdm(trainloader, desc = "Training task {}, Epoch {}/{}".format(self.task_idx, epoch + 1, self.epochs)):
        
        inputs = inputs.to(device)
        labels = labels.to(device)
        labels = labels % self.num_outputs[self.task_idx - 1]

        optimizer.zero_grad()
        outputs = model(inputs)

        loss = self.lossFunc(outputs, labels)
        loss.backward()
        optimizer.step()

    self.eval(self.task_idx, testloader)
  
  def eval(self, task_idx, testloader):
    model = self.models[task_idx - 1]
    model.eval()
    model.to(device)

    eval_total = 0
    eval_correct = 0
    for inputs, labels in tqdm(testloader, desc = "Evaluating Task {}".format(task_idx)):
      inputs = inputs.to(device)
      labels = labels.to(device)
      labels = labels % self.num_outputs[task_idx - 1]

      outputs = model(inputs)

      _, predictions = torch.max(outputs, dim = 1)
      eval_total += labels.shape[0]
      eval_correct += int((predictions == labels).sum()) 
    print("Evaluation Accucary on Task {} : {}".format(task_idx, eval_correct / eval_total))
  

def main():

  args = parser.parse_args()

  #get the loaders for the dataset and set the hyperparameters for it
  if args.dataset == 'cifar100':
    #cifar100 loaders and hyperparameters
    trainloaders, testloaders = get_cifar100()
    num_outputs = 20
    num_tasks = 5
    lr = 1e-3
    epochs = 10
  elif args.dataset == 'pmnist':
    #pmnist loaders and hyperparameters
    trainloaders, testloaders = get_permuted_mnist()
    num_outputs = 10
    num_tasks = 5
    lr = 1e-3
    epochs = 5
  elif args.dataset == 'flowers102':
    #flowers102 loaders and hyperparameters
    trainloaders, testloaders = get_flowers102()
    num_outputs = 17
    num_tasks = 6
    lr = 1e-3
    epochs = 20
  elif args.dataset == 'diffDatasets':
    #diffDatasets loaders and hyperparameters
    trainloaders, testloaders = get_diff_datasets()
    num_outputs = [10, 100, 102]
    num_tasks = 3
    task_lr = [1e-3, 1e-3, 1e-4]
    epochs = 10
  else:
    return


  indNets = IndivindualNets(num_tasks, num_outputs, epochs)

  # Training
  for task_idx, (trainloader, testloader) in enumerate(trainloaders, testloaders):
    
    if args.dataset == 'diffDatasets':
      lr = task_lr[task_idx]

    indNets.train(trainloader, testloader, lr)
  
  # Evaluation
  for task_idx, testloader in enumerate(testloaders):
    indNets.eval(task_idx + 1, testloader)

  

if __name__ == '__main__':
  main()