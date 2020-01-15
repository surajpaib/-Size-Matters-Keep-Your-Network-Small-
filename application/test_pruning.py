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
from Pruning import Pruning

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--pruning_perc', type=float, default=0.1)
parser.add_argument('--epochs', type=int, default=25)
parser.add_argument('--prune_iter', type=int, default=2)



args = parser.parse_args()



class ReshapeTransform:
    def __init__(self, new_size):
        self.new_size = new_size

    def __call__(self, img):
        return torch.reshape(img, self.new_size)



# Define Constants
batch_size_train = 64
batch_size_test = 1000
n_epochs = args.epochs
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
  print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    test_loss, correct, len(test_loader.dataset),
    100. * correct / len(test_loader.dataset)))



pruning = Pruning(percentage=0.1)
pruning.set_test_data(next(iter(test_loader)))

for epoch in range(1, n_epochs + 1):
  train(model, optimizer, epoch)
  test(model)  

  if epoch % args.prune_iter == 0:
    
    optimizer, model = pruning.prune_model(optimizer, model, pruning.layer_conductance_pruning)




