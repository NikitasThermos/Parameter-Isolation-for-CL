import torch
import torch.nn as nn

from blocks import *


class ProgColumn(nn.Module):
  """
  One of the PNN's columns. Outputs are stored to be available for next columns.
  """

  def __init__(self, colID, blockList, parentCols = []):
    super().__init__()
    self.colID = colID
    self.isFrozen = False
    self.parentCols = parentCols
    self.blocks = nn.ModuleList(blockList)
    self.numRows = len(blockList)
    self.lastOutputList = []
  
  def freeze(self, unfreeze = False):
    if not unfreeze: 
      self.isFrozen = True
      for param in self.parameters(): param.requires_grad = False
    else:
      self.isFrozen = False
      for param in self.parameters: param.requires_grad = True
  
  def forward(self, input):
    outputs = []
    x = input
    for row, block in enumerate(self.blocks):
      currOutput = block.runBlock(x)
      if row == 0 or len(self.parentCols) < 1 or not block.isLateralized():
        y = block.runActivation(currOutput)
      else: 
        for c, col in enumerate(self.parentCols):
          currOutput += block.runLateral(c, col.lastOutputList[row - 1])
        y = block.runActivation(currOutput)
      outputs.append(y)
      x = y
    self.lastOutputList = outputs
    return outputs[-1]



class ProgColumnGenerator:
  """ Class that generates automatically new columns with the same architecture """

  def generatorColumn(self, parentCols, msg = None):
    raise NotImplementedError
  


class Resnet18Generator(ProgColumnGenerator):
  """ ResNet-18 architecture columns """

  def __init__(self, num_classes):
    self.ids = 0
    self.num_classes = num_classes
  
  def generateColumn(self, parentCols, msg = None):
    cols = []
    cols.append(ProgConv2DBNBlock(3, 64, 7, 0, activation = nn.ReLU(), layerArgs = {"stride" : 2, "padding" : 3, "bias" : False}))
    cols.append(ProgMaxPool(2, 2))

    cols.append(ProgResnetBlock(64, 64, 3, len(parentCols), activation = nn.ReLU()))
    cols.append(ProgResnetBlock(64, 64, 3, len(parentCols), activation = nn.ReLU()))

    #downsample
    ds1 = self.downsample = nn.Sequential(
      nn.Conv2d(64, 128, kernel_size=1, stride= 2, bias=False),
      nn.BatchNorm2d(128),
    )

    cols.append(ProgResnetBlock(64, 128, 3, len(parentCols), activation = nn.ReLU(), stride = 2, downsample = ds1))
    cols.append(ProgResnetBlock(128, 128, 3, len(parentCols), activation = nn.ReLU()))

    #downsample
    ds2 = self.downsample = nn.Sequential(
      nn.Conv2d(128, 256, kernel_size=1, stride= 2, bias=False),
      nn.BatchNorm2d(256),
    )

    cols.append(ProgResnetBlock(128, 256, 3, len(parentCols), activation = nn.ReLU(), stride = 2, downsample = ds2))
    cols.append(ProgResnetBlock(256, 256, 3, len(parentCols), activation = nn.ReLU()))

    #downsample
    ds3 = self.downsample = nn.Sequential(
      nn.Conv2d(256, 512, kernel_size=1, stride= 2, bias=False),
      nn.BatchNorm2d(512),
    )

    cols.append(ProgResnetBlock(256, 512, 3, len(parentCols), activation = nn.ReLU(), stride = 2, downsample = ds3))
    cols.append(ProgResnetBlock(512, 512, 3, len(parentCols), activation = nn.ReLU()))
   
    cols.append(ProgAdaptiveAvgPool(1, 1))
    cols.append(ProgLambda(lambda x : torch.flatten(x, start_dim = 1)))
    cols.append(ProgDenseBlock(512, self.num_classes, len(parentCols), activation = None))

    return ProgColumn(self.__genID(), cols, parentCols = parentCols)
  
  def __genID(self):
    id = self.ids
    self.ids += 1
    return id 