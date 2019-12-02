#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
sys.path.append('framework')
from NetworkClass import Network
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset
import torchvision
import torch
import torch.nn as nn
import copy
import time
import argparse


parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--pruning_perc', type=float, default=0.1)
parser.add_argument('--epochs', type=int, default=25)
parser.add_argument('--prune_iter', type=int, default=5)



args = parser.parse_args()



class ReshapeTransform:
    def __init__(self, new_size):
        self.new_size = new_size

    def __call__(self, img):
        return torch.reshape(img, self.new_size)



# Define Constants
batch_size_train = 64
batch_size_test = 1000
n_epochs = int(float(args.epochs)/args.prune_iter)
learning_rate = 0.01
momentum = 0.5
log_interval = 10
random_seed = 1




# Model Definition
model_dict = {
        "network":{
            'input_layer': {
                "units": 784,
                
                },
            'hidden_layer': [{
                    "units": 500, 
                    "activation": "relu",
                    "type": "Linear"
                }, 
                {
                    "units": 300, 
                    "activation": "relu",
                    "type": "Linear"

                }],
            'output_layer': {
                "units": 10,
                "activation": "softmax",
                "type": "Linear"
                }
        }
    }

model = Network(model_dict)



for (layer, param) in enumerate(model.parameters()):
    print("Layer {} , Parameters: {}".format(layer, param.shape))




# Load Datasets

train_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('../data/', train=True, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,)), ReshapeTransform((-1,))
                             ])),
  batch_size=batch_size_train, shuffle=True)


test_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('../data/', train=False, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,)), ReshapeTransform((-1,))
                             ])),
  batch_size=batch_size_test, shuffle=True)



# Define Loss and Optimizer

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)



train_losses = []
train_counter = []
test_losses = []
test_counter = [i*len(train_loader.dataset) for i in range(n_epochs + 1)]




def train(mod, optim, epoch):
  mod.train()
  for batch_idx, (data, target) in enumerate(train_loader):
    optim.zero_grad()
    output = mod(data)
    loss = criterion(output, target)
    loss.backward()
    optim.step()
    if batch_idx % log_interval == 0:
      print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        epoch, batch_idx * len(data), len(train_loader.dataset),
        100. * batch_idx / len(train_loader), loss.item()))
      train_losses.append(loss.item())
      train_counter.append(
        (batch_idx*64) + ((epoch-1)*len(train_loader.dataset)))



def test(mod):
  mod.eval()
  test_loss = 0
  correct = 0
  with torch.no_grad():
    for data, target in test_loader:
      output = mod(data)
      test_loss += criterion(output, target).item()
      pred = output.data.max(1, keepdim=True)[1]
      correct += pred.eq(target.data.view_as(pred)).sum()
  test_loss /= len(test_loader.dataset)
  test_losses.append(test_loss)
  print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    test_loss, correct, len(test_loader.dataset),
    100. * correct / len(test_loader.dataset)))


test(model)




import numpy as np
def prune_neurons(nn_model, pruning_perc=0.1):
    print("Pre-Pruning/n")
    for (layer, param) in enumerate(nn_model.parameters()):
        print("Layer {} , Parameters: {}".format(layer, param.shape))    
        
    # 1 neuron is pruned from each layer based on the minimum L1 norm.
    neurons_to_prune = []
    for p in nn_model.parameters():
            if len(p.data.size()) != 1:
                normed_weights = p.data.abs()
                l1_norm_layer = []
                for neuron_idx in range(normed_weights.shape[0]):
                    l1_norm_layer.append(torch.sum(normed_weights[neuron_idx, :]).item())
                    
                threshold = np.percentile(np.array(l1_norm_layer), pruning_perc * 100)
                prune_idx = np.argwhere(np.array(l1_norm_layer) > threshold).flatten()
           
                neurons_to_prune.append(prune_idx)

                
    # Modify the model parameters to update the shape of the network after pruning
    param_list = list(nn_model.parameters())
    
    neurons_to_prune = [val for val in neurons_to_prune for _ in (0, 1)]
    for i, neuron_idx in enumerate(neurons_to_prune):
        idx_weights = param_list[i]
        if i < len(param_list) - 2:
            print(idx_weights.shape)
            y = idx_weights[neuron_idx]
            if i > 1 and len(idx_weights.shape)> 1:
                y = y[:, neurons_to_prune[i-1]]
        elif i > 1 and len(idx_weights.shape) > 1:
            y = idx_weights[:, neurons_to_prune[i-1]]
            
        else:
            y = idx_weights
            
        idx_weights.data = y
       
    print("Post Pruning /n")
    for (layer, param) in enumerate(nn_model.parameters()):
        print("Layer {} , Parameters: {}".format(layer, param.shape))   
        
    return nn_model


# In[75]:


def prune_train(model, optimizer, pruning_perc=0.1):
    new_model = copy.deepcopy(model)
    new_optimizer = copy.deepcopy(optimizer)
    
    # Get new model after pruning
    new_model = prune_neurons(new_model, pruning_perc=pruning_perc)
    new_model_dict = copy.deepcopy(model_dict)
    
    
    for i, layer in enumerate(model_dict['network']['hidden_layer']):
        if i == 0:

            new_model_dict['network']['hidden_layer'][i]['units'] = new_model.state_dict()['input_layer.weight'].shape[0]
        else:
            new_model_dict['network']['hidden_layer'][i]['units'] = new_model.state_dict()['hidden_layers.{}.weight'.format(i-1)].shape[0]

    # Create new model based on updated network definition
    updated_model = new_model
    # Load previously trained parameters into the new model
    # updated_model.load_state_dict(new_model.state_dict())
    
    # Update Optimizers and set state from the previous optimizer
    criterion = nn.CrossEntropyLoss()
    updated_optimizer = torch.optim.SGD(updated_model.parameters(), lr=learning_rate)
    updated_optimizer.load_state_dict(new_optimizer.state_dict())
    
    # Train and test the model
    for epoch in range(1, n_epochs + 1):
      train(updated_model, updated_optimizer, epoch)
      test(updated_model)
        
    return updated_model, updated_optimizer

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# In[76]:


# Pruning every 3 epochs
if torch.cuda.is_available():
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()

else:
    start = time.time()
    
for epoch in range(1, n_epochs + 1):
  train(model, optimizer, epoch)
  test(model)  

print("Model Parameters: ", count_parameters(model))

# Perform Pruning 5 times in succession
for i in range(args.prune_iter-1):
    model, optimizer = prune_train(model, optimizer, 0.2)
    print("Model Parameters: ", count_parameters(model))

if torch.cuda.is_available():
    end.record()
    torch.cuda.synchronize()
    print("Total Training Time: ", start.elapsed_time(end))
else:
    end = time.time()
    print("Total Training Time: ", end - start)

print("Model Parameters: ", count_parameters(model))





