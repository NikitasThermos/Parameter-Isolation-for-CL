
import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SparsePruner(object):
  """Performs pruning on the given model"""

  def __init__(self, model, prune_perc, previous_masks, train_bias = False, train_bn = False):
    self.model = model 
    self.prune_perc = prune_perc
    self.train_bias = train_bias
    self.train_bn = train_bn

    self.current_masks = None
    self.previous_masks = previous_masks
    valid_key = list(previous_masks.keys())[0]
    self.current_dataset_idx = previous_masks[valid_key].max()
  
  def pruning_mask(self, weights, previous_mask, layer_idx):
    """Ranks weights by magnitude. Sets all below kth to 0.Returns pruned mask"""

    # Select all prunable weights, i.e. belonging to the current dataset.
    previous_mask = previous_mask.to(device)
    prunable_weights = weights[previous_mask.eq(self.current_dataset_idx.to(device))]
    abs_prunable = prunable_weights.abs()
    cutoff_rank = round(self.prune_perc * prunable_weights.numel())
    cutoff_value = abs_prunable.view(-1).cpu().kthvalue(cutoff_rank)[0].item()

    #Remove those weights which are below cutoff and belong to
    #the current dataset that we are training for.
    remove_mask = weights.abs().le(cutoff_value) * previous_mask.eq(self.current_dataset_idx.to(device))

    
    previous_mask[remove_mask.eq(1)] = 0
    mask = previous_mask
    print('Layer #%d, pruned %d/%d (%.2f%%) (Total in layer: %d)' %
              (layer_idx, mask.eq(0).sum(), prunable_weights.numel(),
               100 * mask.eq(0).sum() / prunable_weights.numel(), weights.numel()))
    return mask
  
  def prune(self):
    """Gets prunning mask for each layer, based on previous masks.
       Sets the self.current_masks to the computed pruning masks
    """
    print('Pruning fo dataset idx: %d' %(self.current_dataset_idx))
    assert not self.current_masks, 'Current mask is not empty which means that pruning is already done for this task '
    self.current_masks = {}

    print('Pruning each layer by removing %.2f%% of values' % (100 * self.prune_perc))

    for module_idx, module in enumerate(self.model.shared.modules()):
      if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
        mask = self.pruning_mask(module.weight.data, self.previous_masks[module_idx], module_idx)
        self.current_masks[module_idx] = mask.to(device)
        #Set pruned weights to 0
        weight = module.weight.data
        weight[self.current_masks[module_idx].eq(0)] = 0.0

  def make_grads_zero(self):
    """Sets grads of fixed weights to 0."""
    assert self.current_masks
    for module_idx, module in enumerate(self.model.shared.modules()):
      if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
        layer_mask = self.current_masks[module_idx]

        # Set grads of all weights not belonging to current dataset to 0.
        if module.weight.grad is not None:
          module.weight.grad.data[layer_mask.ne(self.current_dataset_idx.to(device))] = 0
          if not self.train_bias:
            #Biases are fixed
            if module.bias is not None:
              module.bias.grad.fill_(0)
      
      elif 'BatchNorm' in str(type(module)):
        # Set grads of batchnorm to 0
        if not self.train_bn:
          module.weight.grad.data.fill_(0)
          module.bias.grad.fill_(0)
  
  def make_pruned_zero(self):
    """Makes pruned weight 0"""
    assert self.current_masks

    for module_idx, module in enumerate(self.model.shared.modules()):
      if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
        layer_mask = self.current_masks[module_idx]
        module.weight.data[layer_mask.eq(0)] = 0.0

  def apply_mask(self, dataset_idx):
    """Applies mask for a specific task for evaluation"""
    for module_idx, module in enumerate(self.model.shared.modules()):
      if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
        weight = module.weight.data
        mask = self.previous_masks[module_idx].to(device)
        weight[mask.eq(0)] = 0.0
        weight[mask.gt(dataset_idx)] = 0.0
  
  def restore_biases(self, biases):
    """Use the given biases to replace existing biases"""
    for module_idx, module in enumerate(self.model.shared.modules()):
      if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
        if module.bias is not None:
          module.bias.data.copy_(biases[module_idx])

  def get_biases(self):
    """Gets a copy of the current biases"""
    biases = {}
    for module_idx, module in enumerate(self.model.shared.modules()):
      if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
        if module.bias is not None:
          biases[module_idx] = module.bias.data.clone()
    return biases
  
  def make_finetuning_mask(self):
    """Turns previously pruned weights (id = 0) into trainable weights for the current dataset (id = current_dataset_id)"""
    assert self.previous_masks
    self.current_dataset_idx += 1
    print("Current Dataset Idx:", self.current_dataset_idx)

    for module_idx, module in enumerate(self.model.shared.modules()):
      if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
        mask = self.previous_masks[module_idx]
        mask[mask.eq(0)] = self.current_dataset_idx
    
    self.current_masks = self.previous_masks