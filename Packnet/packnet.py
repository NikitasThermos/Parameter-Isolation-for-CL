
import argparse
from Datasets.datasets import *
from networks import *

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

from pruner import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', choices = ['pmnist', 'cifar100', 'flowers102', 'diffDatasets'], required = True, help = 'The dataset to be used for the experiment')
packnet_path = 'packnet_weights.pth'



class Packnet(object):

  def __init__(self, args, model, previous_masks, trainloader, testloader):
    self.args = args 
    self.model = model

    self.train_data_loader = trainloader
    self.test_data_loader = testloader

    self.criterion = nn.CrossEntropyLoss()

    self.pruner = SparsePruner(
        self.model, self.args['prune_perc_per_layer'], previous_masks, 
        self.args['train_biases'], self.args['train_bn'])

  def train(self, epochs, optimizer):

    self.model.to(device)

    # Train batch normalization from the first task. Next tasks use the same stats. !!!!!!
    if self.pruner.current_dataset_idx == 1:
      self.model.train()
    else:
      self.model.train_nobn()

    for epoch in range(epochs):
      for inputs, label in tqdm(self.train_data_loader, desc = 'Epoch {}/{}'.format(epoch + 1, epochs)):
        
        inputs = inputs.to(device)
        label = label.to(device)
        label = label % self.args['num_outputs']

        self.model.zero_grad()

        #Do forward-backward
        output = self.model(inputs)
        self.criterion(output, label).backward()

        #Set fixed param grads to 0.
        self.pruner.make_grads_zero()

        optimizer.step()

        #Set pruned weights to 0.
        self.pruner.make_pruned_zero()
        

    self.eval(self.pruner.current_dataset_idx)
    print('Finished Training...')
    print('-' * 16)


  def eval(self, dataset_idx, biases = None):

    self.pruner.apply_mask(dataset_idx)
    if biases is not None:
      self.pruner.restore_biases(biases)
    
    self.model.eval()
    eval_total = 0
    eval_correct = 0
    with torch.no_grad():
      for inputs, label in tqdm(self.test_data_loader, desc = 'Evaluation on task {}'.format(dataset_idx)):
        inputs = inputs.to(device)
        label = label.to(device)
        label = label % self.args['num_outputs']

        output = self.model(inputs)

        _, predictions = torch.max(output, dim = 1)
        eval_total += label.shape[0]
        eval_correct += int((predictions == label).sum())
      print("Valdation Accuracy: {:.4f}"
       .format(eval_correct / eval_total))

  def prune(self):
    print('Pre-prune evaluation:')
    self.eval(self.pruner.current_dataset_idx)

    self.pruner.prune()
    self.check(True)

    print("\nPost-prune evaluation:")
    self.eval(self.pruner.current_dataset_idx)
    
    if self.args['post_prune_epochs']:
      print('Doing post prune training...')
  
      optimizer = optim.Adam(self.model.parameters(), lr = self.args['lr'])
      self.train(self.args['post_prune_epochs'], optimizer)
    
    print('-' * 16)
    print('Pruning summary:')
    self.check(True)
    

  def check(self, verbose = True):
    """Checks what percent of each layer is pruned"""
    print('Checking Network...')
    for layer_idx, module in enumerate(self.model.shared.modules()):
      if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
        weight = module.weight.data
        num_params = weight.numel()
        num_zero = weight.view(-1).eq(0).sum()
        if verbose:
           print('Layer #%d: Pruned %d/%d (%.2f%%)' %
                 (layer_idx, num_zero, num_params, 100 * num_zero / num_params))
  
  def save_model(self, savename):
        """Saves model to file."""
        base_model = self.model

        # Prepare the ckpt.
        ckpt = {
            'args': self.args,
            'previous_masks': self.pruner.current_masks,
            'model': base_model,
        }

        # Save to file.
        torch.save(ckpt, savename)
  
  def train_task(self):
    self.pruner.make_finetuning_mask()
    optimizer = optim.Adam(self.model.parameters(), lr = self.args['lr'])

    #Perform finetunning
    self.train(self.args['finetune_epochs'], optimizer)


def main(): 
  args = parser.parse_args()

   #get the loaders for the dataset and set the hyperparameters for it
  if args.dataset == 'cifar100':
    #cifar100 loaders and hyperparameters
    trainloaders, testloaders = get_cifar100()
    
    train_args = {"num_outputs" : 20, "lr" : 1e-3,  "finetune_epochs" : 7, "dataset" : "CIFAR100_", 
              "prune_perc_per_layer" : 0.7, "train_biases" : False, "train_bn" : False}

    prune_args = {"num_outputs" : 20, "lr" : 1e-4, "dataset" : "CIFAR100_", 
              "prune_perc_per_layer" : 0.7, "post_prune_epochs" : 3, "train_biases" : False, "train_bn" : False}
  
  elif args.dataset == 'pmnist':
    #pmnist loaders and hyperparameters
    trainloaders, testloaders = get_permuted_mnist()
    
    train_args = {"num_outputs" : 10, "lr" : 1e-3,  "finetune_epochs" : 3, "dataset" : "PMNIST_", 
              "prune_perc_per_layer" : 0.7, "train_biases" : False, "train_bn" : False}

    prune_args = {"num_outputs" : 10, "lr" : 1e-4, "dataset" : "PMNIST_", 
              "prune_perc_per_layer" : 0.7, "post_prune_epochs" : 2,  "train_biases" : False, "train_bn" : False}
  
  elif args.dataset == 'flowers102':
    #flowers102 loaders and hyperparameters
    trainloaders, testloaders = get_flowers102()
    
    train_args = {"num_outputs" : 17, "lr" : 1e-3,  "finetune_epochs" : 15, "dataset" : "Flowers_", 
              "prune_perc_per_layer" : 0.7, "train_biases" : False, "train_bn" : False}

    prune_args = {"num_outputs" : 17, "lr" : 1e-4, "dataset" : "Flowers_", 
              "prune_perc_per_layer" : 0.7, "post_prune_epochs" : 5,  "train_biases" : False, "train_bn" : False}

  elif args.dataset == 'diffDatasets':
    #diffDatasets loaders and hyperparameters
    trainloaders, testloaders = get_diff_datasets()

    train_args = {"lr" : 1e-3, "finetune_epochs" : 7,  "prune_perc_per_layer" : 0.7, "train_biases" : False, "train_bn" : False}
    prune_args = {"lr" : 1e-4, "post_prune_epochs": 3,  "prune_perc_per_layer" : 0.7, "train_biases" : False, "train_bn" : False}

    datasets = ['MNIST', 'CIFAR100', 'FLOWERS102']
    num_classes = [10, 100, 102]
    
  else:
    return
  
  #New PackNet model
  model = PacknetResNet()
  previous_masks = {}

  # Filling masks with 0s, so every parameter is available for the first task
  for module_idx, module in enumerate(model.shared.modules()):
    if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
      mask = torch.ByteTensor(module.weight.data.size()).fill_(0)
      mask = mask.to(device)
      previous_masks[module_idx] = mask
  
  #Training
  task_idx = 1
  for trainloader, testloader in zip(trainloaders, testloaders):
    
    #if the experiment uses 'diffDatasets' then take the number of classes for the new task
    if args.dataset == 'diffDatasets':
      train_args['num_outputs'] = num_classes[task_idx - 1]
      train_args['dataset'] = datasets[task_idx - 1]

      prune_args['num_outputs'] = num_classes[task_idx - 1]
      prune_args['dataset'] = datasets[task_idx - 1]

    #train
    packnet = Packnet(train_args, model, previous_masks, trainloader, testloader)
    model.add_dataset(train_args['dataset'] + str(task_idx), train_args['num_outputs'])
    model.set_dataset(train_args['dataset'] + str(task_idx))
    model.to(device)
    packnet.train_task()

    #prune
    previous_masks = packnet.pruner.current_masks
    packnet = Packnet(prune_args, model, previous_masks, trainloader, testloader)
    packnet.prune()

    task_idx += 1

    previous_masks = packnet.pruner.current_masks

  packnet.save_model(packnet_path)


  #evaluation
  eval_idx = 1
  for trainloader , testloader in zip(trainloaders, testloaders):
    ckpt = torch.load(packnet_path)
    model = ckpt['model']
    previous_masks = ckpt['previous_masks']

    if args.dataset == 'diffDatasets':
      train_args['num_outputs'] = num_classes[eval_idx - 1]
      train_args['dataset'] = datasets[eval_idx - 1]

    model.set_dataset(train_args['dataset'] + str(task_idx))
    model.to(device)

    
    packnet = Packnet(train_args, model, previous_masks, trainloader, testloader)
    packnet.eval(task_idx)

    eval_idx += 1
  


if __name__ == '__main__':
  main()