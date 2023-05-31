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

class Finetune:
  def __init__(self, num_outputs, epochs, lr):

    self.num_outputs = num_outputs

    self.model = models.resnet34(weights = None)
    self.model.fc = nn.Linear(self.model.fc.in_features, self.num_outputs)

    self.epochs = epochs
    self.lr = lr
    self.lossFunc = nn.CrossEntropyLoss()
    self.task_idx = 0

  def train(self, trainloader, testloader):
    self.task_idx += 1
    self.model.train()
    self.model.to(device)
    optimizer = torch.optim.Adam(self.model.parameters(), lr = self.lr)

    for epoch in range(self.epochs):      
      for inputs, labels in tqdm(trainloader, desc = "Training task {}, Epoch {}/{}".format(self.task_idx, epoch + 1, self.epochs)):

        inputs = inputs.to(device)
        labels = labels.to(device)
        labels = labels % self.num_outputs

        optimizer.zero_grad()

        outputs = self.model(inputs)
        loss = self.lossFunc(outputs, labels)

        loss.backward()
        optimizer.step()
        
    self.eval(self.task_idx, testloader)
  
  def eval(self, task_idx, testloader):
    eval_total = 0
    eval_correct = 0
    self.model.eval()
    self.model.to(device)
    for inputs, labels in tqdm(testloader, desc = "Evaluating Task {}".format(task_idx)):
      
      inputs = inputs.to(device)
      labels = labels.to(device)
      labels = labels % self.num_outputs

      outputs = self.model(inputs)

      _, predictions = torch.max(outputs, dim = 1)
      eval_total += labels.shape[0]
      eval_correct += int((predictions == labels).sum()) 
    print("Evaluation Accucary on Task {} : {}".format(task_idx, eval_correct/eval_total))



def main():
  
  #agr parse the dataset user gave.
  args = parser.parse_args()

  #get the loaders for the dataset and set the hyperparameters for it
  if args.dataset == 'cifar100':
    #cifar100 loaders and hyperparameters
    trainloaders, testloaders = get_cifar100()
    num_outputs = 20
    lr = 1e-3
    epochs = 10
  elif args.dataset == 'pmnist':
    #pmnist loaders and hyperparameters
    trainloaders, testloaders = get_permuted_mnist()
    num_outputs = 10
    lr = 1e-3
    epochs = 5
  elif args.dataset == 'flowers102':
    #flowers102 loaders and hyperparameters
    trainloaders, testloaders = get_flowers102()
    num_outputs = 17
    lr = 1e-3
    epochs = 20
  elif args.dataset == 'diffDatasets':
    #diffDatasets loaders and hyperparameters
    trainloaders, testloaders = get_diff_datasets()
    num_outputs = 102
    lr = 1e-3
    epochs = 10
  else:
    return
  
  #Train
  finetune = Finetune(num_outputs, epochs, lr)
  for trainloader, testloader in zip(trainloaders, testloaders):
    finetune.train(trainloader, testloader)
  
  #Evaluation
  for task_idx, testloader in enumerate(testloaders):
    finetune.eval(task_idx + 1, testloader)


if __name__ == '__main__':
  main()