
import argparse
from Datasets.datasets import *
from column import *

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', choices = ['pmnist', 'cifar100', 'flowers102', 'diffDatasets'], required = True, help = 'The dataset to be used for the experiment')

class Pnn(nn.Module):
  """ The main PNN class """
  def __init__(self, colGen = None):
    super().__init__()
    self.columns = nn.ModuleList()
    self.numCols = 0
    self.colMap = dict()
    self.colGen = colGen
  
  def addColumn(self, col = None, msg = None):
    if not col:
      parents = [colRef for colRef in self.columns]
      col = self.colGen.generateColumn(parents, msg)
    col = col.to(device)
    self.columns.append(col)
    self.colMap[col.colID] = self.numCols
    self.numCols += 1
    return col.colID
  
  def freezeColumn(self, id):
    col = self.columns[self.colMap[id]]
    col.freeze()
  
  def freezeAllColumns(self):
    for col in self.columns:
      col.freeze()
  
  def unfreezeColumn(self, id):
    col = self.columns[self.colMap[id]]
    col.freeze(unfreeze = True)
  
  def unfreezeAllColumns(self):
    for col in self.columns:
      col.freeze(unfreeze = True)
  
  def forward(self, id, x):
    colToOutput = self.colMap[id]
    for i, col in enumerate(self.columns):
      y = col(x)
      if i == colToOutput:
        return y 

  def getColumn(self, id):
    col = self.columns[self.colMap[id]]
    return col

  def begin_task(self):
   return self.addColumn()
  
  def end_task(self):
    self.freezeAllColumns()
  
  def trainTask(self, lr,  epochs, trainloader, testloader, num_outputs = 10):

    col = self.addColumn()
    self.columns[-1].train() #train only the last column

    optimizer = torch.optim.Adam(self.parameters(), lr = lr)

    for epoch in range(epochs):
      for inputs, labels in tqdm(trainloader, desc = 'Training Column {}, Epoch {}/{}'.format(col, epoch + 1, epochs)):
        inputs = inputs.to(device)
        labels = labels.to(device)
        labels = labels % num_outputs
       
        output = self(col, inputs)
        loss = F.cross_entropy(output, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    self.evalTask(col, testloader, num_outputs)

  def evalTask(self, task_idx, testloader, num_outputs = 10):
    self.eval()
    total = 0
    correct = 0
    with torch.no_grad():
      for inputs, labels in tqdm(testloader, desc = 'Evaluating Task {}'.format(task_idx)):
        inputs = inputs.to(device)
        labels = labels.to(device)
        labels = labels % num_outputs
        output = self(task_idx, inputs)

        _, predictions = torch.max(output, dim = 1)
        total += labels.shape[0]
        correct += int((predictions == labels).sum()) 

    print("Col: {}, Testing Accuracy: {:.4f}".format(task_idx, correct/total))



def main() : 
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
    task_outputs = [10, 100, 102]
    lr = 1e-3
    epochs = 10
  else:
    return
  

  task_idx = 0
  # If dataset is 'diffDatasets' then take the number of classes of the first task
  if args.dataset == 'diffDatasets':
    num_outputs = task_outputs[task_idx]
  
  model = Pnn(colGen = Resnet18Generator(num_classes = num_outputs))
  model.to(device)

  #Training
  for trainloader, testloader in zip(trainloaders, testloaders):
    model.trainTask(lr, epochs, trainloader, testloader, num_outputs = num_outputs)
    model.freezeAllColumns()

    #Change the number of outputs for the next classifier to match the number of classes of the new task.
    if args.dataset == 'diffDatasets':
      if task_idx < 2:
        task_idx += 1
        model.colGen.num_classes = task_outputs[task_idx]
  
  #Evaluation
  for task_idx, testloader in enumerate(testloaders):
    model.evalTask(task_idx, testloader, num_outputs = num_outputs)


if __name__ == '__main__':
  main()