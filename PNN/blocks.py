import torch
import torch.nn as nn



class ProgBlock(nn.Module):
  def runBlock(self, x):
    raise NotImplementedError
  
  def runLateral(self, i, x):
    raise NotImplementedError
  
  def runActivation(self, x):
    raise NotImplementedError
  
  def isLateralized(self):
    return True

class ProgInertBlock(ProgBlock):
  def isLateralized(self):
    return False


class ProgDenseBlock(ProgBlock):
  """ A ProgBlock containing a single fully connected layer(nn.Linear) """

  def __init__(self, inSize, outSize, numLaterals, activation = nn.ReLU()):
    super().__init__()
    self.numLaterals = numLaterals
    self.inSize = inSize 
    self.outSize = outSize
    self.module = nn.Linear(inSize, outSize)
    self.laterals = nn.ModuleList([nn.Linear(inSize, outSize) for _ in range(numLaterals)])
    if activation is None: self.activation = (lambda x : x)
    else:                  self.activation = activation

  def runBlock(self, x):
    return self.module(x)

  def runLateral(self, i, x):
    lat = self.laterals[i]
    return lat(x)
  
  def runActivation(self, x):
    return self.activation(x)


class ProgDenseBNBlock(ProgBlock):
  """ A ProgBlock containing a single fully connected layer and batch normalization """
  def __init__(self, inSize, outSize, numLaterals, activation = nn.ReLU(), bnArgs = dict()):
    super().__init__()
    self.numLaterals = numLaterals
    self.inSize = inSize
    self.outSize = outSize
    self.module = nn.Linear(inSize, outSize)
    self.moduleBN = nn.BatchNorm1d(outSize, **bnArgs)
    self.laterals = nn.ModuleList([nn.Linear(inSize, outSize) for _ in range(numLaterals)])
    self.lateralBNs = nn.ModuleList([nn.BatchNorm1d(outSize, **bnArgs) for _ in range(numLaterals)])
    if activation is None: self.activation = (lambda x : x) 
    else:                  self.activation = activation
  
  def runBlock(self, x):
    return self.moduleBN(self.module(x))
  
  def runLateral(self, i, x):
    lat = self.laterals[i]
    bn = self.lateralBNs[i]
    return bn(lat(x))
  
  def runActivation(self, x):
    return self.activation(x)


class ProgConv2DBlock(ProgBlock):
  """ A ProgBlock containing a single Conv2D layer """

  def __init__(self, inSize, outSize, kernelSize, numLaterals, activation = nn.ReLU(), layerArgs = dict()):
    super().__init__()
    self.numLaterals = numLaterals
    self.inSize = inSize
    self.outSize = outSize
    self.kernSize = kernelSize
    self.module = nn.Conv2d(inSize, outSize, kernelSize, **layerArgs)
    self.laterals = nn.ModuleList([nn.Conv2d(inSize, outSize, kernelSize, **layerArgs) for _ in range(numLaterals)])
  
    if activation is None: self.activation = (lambda x : x)
    else:                  self.activation = activation
  
  def runBlock(self, x):
    return self.module(x)

  def runLateral(self, i, x):
    lat = self.laterals[i]
    return lat(x)
  
  def runActivation(self, x):
    return self.activation(x)

class ProgConv2DBNBlock(ProgBlock):
  """ A ProgBlock contaiing a single Conv2d and Batch Normalization """

  def __init__(self, inSize, outSize, kernelSize, numLaterals, activation = nn.ReLU(), layerArgs = dict(), bnArgs = dict()):
        super().__init__()

        self.numLaterals = numLaterals

        self.inSize = inSize
        self.outSize = outSize
        self.kernSize = kernelSize
        self.module = nn.Conv2d(inSize, outSize, kernelSize, **layerArgs)

        self.moduleBN = nn.BatchNorm2d(outSize, **bnArgs)
        self.laterals = nn.ModuleList([nn.Conv2d(inSize, outSize, kernelSize, **layerArgs) for _ in range(numLaterals)])
        self.lateralBNs = nn.ModuleList([nn.BatchNorm2d(outSize, **bnArgs) for _ in range(numLaterals)])

        if activation is None:   self.activation = (lambda x: x)
        else:                    self.activation = activation


  def runBlock(self, x):
        return self.moduleBN(self.module(x))

  def runLateral(self, i, x):
        lat = self.laterals[i]
        bn = self.lateralBNs[i]
        return bn(lat(x))

  def runActivation(self, x):
        return self.activation(x)


class ProgResnetBlock(ProgBlock):
  """ A ProgBlock with two Conv2d layers with batch normalization and a skip connection """

  def __init__(self, inSize, outSize, kernelSize, numLaterals, activation = nn.ReLU(), bnArgs = dict(), stride = 1, expansion = 1,  downsample = None):

      super().__init__()
      self.numLaterals = numLaterals
        
      self.expansion = expansion
      self.downsample = downsample

      self.inSize = inSize
      self.outSize = outSize
      self.kernSize = kernelSize

      self.conv1 = nn.Conv2d(inSize, outSize, kernelSize, stride = stride, padding = 1, bias = False)
      self.bn1 = nn.BatchNorm2d(outSize)

      self.conv2 = nn.Conv2d(outSize, outSize * self.expansion, kernelSize, padding = 1, bias = False)
      self.bn2 = nn.BatchNorm2d(outSize * self.expansion)

      self.laterals = nn.ModuleList([nn.Conv2d(inSize, outSize, kernelSize, stride = stride, padding = 1) for _ in range(numLaterals)])
      self.lateralBNs = nn.ModuleList([nn.BatchNorm2d(outSize) for _ in range(numLaterals)])

      if activation is None:   self.activation = (lambda x: x)
      else:                    self.activation = activation


  def runBlock(self, x):
      input = x
        
      out = self.conv1(x)
      out = self.bn1(out)
      out = self.activation(out)

      out = self.conv2(out)
      out = self.bn2(out)

      if self.downsample is not None:
        input = self.downsample(x)

      out += input

      return out

  def runLateral(self, i, x):
      lat = self.laterals[i]
      bn = self.lateralBNs[i]
      return bn(lat(x))

  def runActivation(self, x):
      return self.activation(x)
  

class ProgLambda(ProgInertBlock):
  """ A ProgBlock with no lateral connections that runs input through lambda functions """

  def __init__(self, module):
    super().__init__()
    self.module = module
  
  def runBlock(self, x):
    return self.module(x)
  
  def runActivation(self, x):
    return x

class ProgMaxPool(ProgInertBlock):
  """ A ProgBlock that runs input through Max Pool  """
  def __init__(self, kernel_size, stride = None,  padding = 0):
    super().__init__()
    self.module = nn.MaxPool2d(kernel_size, stride, padding)

  def runBlock(self, x):
    return self.module(x)
  
  def runActivation(self, x):
    return x

class ProgAdaptiveAvgPool(ProgInertBlock):
  """ A ProgBlock that runs input through Adaptive Average Pool """
  def __init__(self, h, w):
    super().__init__()
    self.module = nn.AdaptiveAvgPool2d((h, w))

  def runBlock(self, x):
    return self.module(x)
  
  def runActivation(self, x):
    return x 