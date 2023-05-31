
import torch
import torch.nn as nn

import torchvision
import torchvision.models as models


class PacknetResNet(nn.Module):
  """ResNet-50"""

  def __init__(self, make_model = True):
    super(PacknetResNet, self).__init__()

    resnet = models.resnet50(weights = None)
    
    self.datasets = [] 
    self.classifiers = nn.ModuleList()
    self.shared = nn.Sequential()
   
    for name, module in resnet.named_children():
      if name != 'fc':
        self.shared.add_module(name, module)
        
    #model.set_dataset() has to be called explicity, else model won't work
    self.classifier = None

  def train_nobn(self, mode = True):
    """Override the default module train"""
    super(PacknetResNet, self).train(mode)

    # Set the BNs to eval mode so that the running means and averages do not update
    for module in self.shared.modules():
      if 'BatchNorm' in str(type(module)):
        module.eval()
  
  def add_dataset(self, dataset, num_outputs):
    """Adds a new dataset and a new classifier to train for the new task. """
    if dataset not in self.datasets:
      print("Adding new dataset : {} with number of outputs : {}".format(dataset, num_outputs))
      self.datasets.append(dataset)
      self.classifiers.append(nn.Linear(2048, num_outputs))

  def set_dataset(self, dataset):
    """Change the active classifier"""
    assert dataset in self.datasets
    self.classifier = self.classifiers[self.datasets.index(dataset)]
    print("Setting dataset : {} with classifier : {}".format(dataset, self.classifier))
  
  def forward(self, x):
    x = self.shared(x)
    x = x.view(x.size(0), - 1)
    x = self.classifier(x)
    return x