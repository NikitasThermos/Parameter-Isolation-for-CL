
import torch
from torch.utils.data import DataLoader

import torchvision
import torchvision.transforms as transforms




def get_cifar100(batch_size = 64):

  # Define the transform to apply to the data
  train_transform = transforms.Compose([
      transforms.RandomHorizontalFlip(),
      transforms.ToTensor(),
      transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])

  test_transform = transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])

  # Load the CIFAR-100 dataset
  train_dataset = torchvision.datasets.CIFAR100('/cifar100', train=True, download=True, transform=train_transform)
  test_dataset = torchvision.datasets.CIFAR100('/cifar100', train=False, download=True, transform=test_transform)

  
  # Define the number of classes per task
  num_classes_per_task = 20
  num_tasks = 5
  
  train_task_loaders = []
  test_task_loaders = []

  for i in range(num_tasks):
    classes = list(range(i*num_classes_per_task, (i+1)*num_classes_per_task))
    train_task_dataset = torch.utils.data.Subset(train_dataset, [j for j in range(len(train_dataset)) if train_dataset[j][1] in classes])
    test_task_dataset = torch.utils.data.Subset(test_dataset, [j for j in range(len(test_dataset)) if test_dataset[j][1] in classes])
    train_loader = DataLoader(train_task_dataset, batch_size = batch_size, shuffle = True, num_workers = 2)
    test_loader = DataLoader(test_task_dataset, batch_size = batch_size, shuffle = False)

    train_task_loaders.append(train_loader)
    test_task_loaders.append(test_loader)
  
  return train_task_loaders, test_task_loaders


class PermuteMNISTTask:
    def __init__(self, permutation):
        self.permutation = permutation

    def __call__(self, x):
        x = x.view(-1, 32 * 32)
        x = x[:, self.permutation]
        return x.view(-1, 32, 32)

def get_permuted_mnist(batch_size = 64):

  # Define random permutations
  permutations = [torch.randperm(32 * 32) for i in range(5)]

  # Create transforms
  tasks = []
  for permutation in permutations:
      tasks.append(transforms.Compose([
          transforms.Grayscale(num_output_channels=3),
          torchvision.transforms.Resize(32),
          transforms.ToTensor(),
          PermuteMNISTTask(permutation)
      ]))              

  train_loaders = []
  test_loaders = []

  # Create tasks    
  for task in tasks:
    train_dataset = torchvision.datasets.MNIST('/mnist', train=True, download=True, transform = task)
    test_dataset = torchvision.datasets.MNIST('/mnist', train=False, download=True, transform = task)

    train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True, num_workers = 2)
    test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle = False)

    train_loaders.append(train_loader)
    test_loaders.append(test_loader)

  return train_loaders, test_loaders


def get_flowers102(batch_size = 16):
  # Create transforms
  train_transform = transforms.Compose([
      transforms.Resize([256, 256]),
      transforms.CenterCrop(224),
      transforms.RandomHorizontalFlip(),
      transforms.ToTensor(),
      transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])

  test_transform = transforms.Compose([
      transforms.Resize([256, 256]),
      transforms.CenterCrop(224),
      transforms.ToTensor(),
      transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])
  
  train_dataset = torchvision.datasets.Flowers102('/flowers102', split="train", download=True, transform=train_transform)
  test_dataset = torchvision.datasets.Flowers102('/flowers102', split="test", download=True, transform=test_transform)

  num_classes_per_task = 17
  num_tasks = 6

  train_task_loaders = []
  test_task_loaders = []
  
  # Create tasks 
  for i in range(num_tasks):
    classes = list(range(i*num_classes_per_task, (i+1)*num_classes_per_task))
    train_task_dataset = torch.utils.data.Subset(train_dataset, [j for j in range(len(train_dataset)) if train_dataset[j][1] in classes])
    test_task_dataset = torch.utils.data.Subset(test_dataset, [j for j in range(len(test_dataset)) if test_dataset[j][1] in classes])
    train_loader = DataLoader(train_task_dataset, batch_size = batch_size, shuffle = True, num_workers = 2)
    test_loader = DataLoader(test_task_dataset, batch_size = batch_size, shuffle = False)

    train_task_loaders.append(train_loader)
    test_task_loaders.append(test_loader)

  return train_task_loaders, test_task_loaders


def get_diff_datasets(batch_size = 32):
  """ Create an experiment with MNIST, CIFAR100 and Flowers102 as different tasks  """

  trainloaders = []
  testloaders = []

  #CIFAR100 and Flowers102 transforms
  train_transform = transforms.Compose([
      transforms.Resize([256, 256]),
      transforms.CenterCrop(224),
      transforms.RandomHorizontalFlip(),
      transforms.ToTensor(),
      transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])
  
  test_transform = transforms.Compose([
      transforms.Resize([256, 256]),
      transforms.CenterCrop(224),
      transforms.ToTensor(),
      transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])

  #MNIST transform
  mnist_train = transforms.Compose([
      transforms.Grayscale(num_output_channels=3),
      train_transform
  ])

  mnist_test = transforms.Compose([
      transforms.Grayscale(num_output_channels=3),
      test_transform
  ])

  # Load MNIST
  mnist_train = torchvision.datasets.MNIST('/mnist', train=True, download=True, transform = mnist_train)
  mnist_test = torchvision.datasets.MNIST('/mnist', train=False, download=True, transform = mnist_test)

  mnist_train_loader = DataLoader(mnist_train, batch_size = batch_size, shuffle = True, num_workers = 2)
  mnist_test_loader = DataLoader(mnist_test, batch_size = batch_size, shuffle = False)

  trainloaders.append(mnist_train_loader)
  testloaders.append(mnist_test_loader)

  #Load CIFAR100
  cifar_train = torchvision.datasets.CIFAR100('/cifar100', train=True, download=True, transform=train_transform)
  cifar_test = torchvision.datasets.CIFAR100('/cifar100', train=False, download=True, transform=test_transform)

  cifar_train_loader = DataLoader(cifar_train, batch_size = batch_size, shuffle = True, num_workers = 2)
  cifar_test_loader = DataLoader(cifar_test, batch_size = batch_size, shuffle = False)

  trainloaders.append(cifar_train_loader)
  testloaders.append(cifar_test_loader)

  # Load Flowers102
  flowers_train = torchvision.datasets.Flowers102('/flowers102', split="train", download=True, transform=train_transform)
  flowers_test = torchvision.datasets.Flowers102('/flowers102', split="test", download=True, transform=test_transform)

  flowers_train_loader = DataLoader(flowers_train, batch_size = batch_size, shuffle = True, num_workers = 2)
  flowers_test_loader = DataLoader(flowers_test, batch_size = batch_size, shuffle = False)
  
  trainloaders.append(flowers_train_loader)
  testloaders.append(flowers_test_loader)

  return trainloaders, testloaders