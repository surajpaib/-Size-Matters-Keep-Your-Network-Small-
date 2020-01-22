#!/usr/bin/env python
# coding: utf-8

# In[2]:


import sys
sys.path.append('../../framework')
sys.path.append('../../application')

from NetworkClass import Network        
import copy
import logging
import string
import random
from copy import deepcopy
import os
import torch
import torchvision
from sklearn.model_selection import KFold

from Experiment import Experiment
from train_utils import ReshapeTransform



loaded = torch.load(sys.argv[1], map_location=torch.device('cpu'))
model_dict = copy.deepcopy(loaded["params"]["model"])

for i, layer in enumerate(loaded["params"]["model"]['network']['hidden_layer']):
    if i == 0:

        model_dict['network']['hidden_layer'][i]['units'] = loaded["state_dict"]['input_layer.weight'].shape[0]
    else:
        model_dict['network']['hidden_layer'][i]['units'] = loaded["state_dict"]['hidden_layers.{}.weight'.format(i-1)].shape[0]

model = Network(model_dict)
model.load_state_dict(loaded["state_dict"])


def randomString(stringLength=10):
    """Generate a random string of fixed length """
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(stringLength))

logging.basicConfig(level=logging.INFO)

if not(os.path.isdir('models')):
  os.mkdir('models')

params_dict = {
  "batch_size_train": 100,
  "learning_rate": 0.01,
  "batch_size_test": 1000,
  "n_epochs": 200
}

seed = 42
uid = randomString(stringLength=6)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
  torch.cuda.manual_seed(seed)
else:
  torch.manual_seed(seed)


params_dict["model"] = model_dict

train_dataset = torchvision.datasets.FashionMNIST('../data/', train=True, download=True,
                                                                  transform=torchvision.transforms.Compose([
                                                                      torchvision.transforms.ToTensor(),
                                                                      ReshapeTransform(
                                                                          (-1,))
                                                                  ]))

test_dataset = torchvision.datasets.FashionMNIST('../data/', train=False, download=True,
                                                                 transform=torchvision.transforms.Compose([
                                                                     torchvision.transforms.ToTensor(),
                                                                     ReshapeTransform(
                                                                         (-1,))
                                                                 ]))

dataset = torch.utils.data.ConcatDataset(
                    [train_dataset, test_dataset])

kf = KFold(n_splits=5, shuffle=True, random_state=seed)
for i_fold, (train_index, test_index) in enumerate(kf.split(dataset)):
    print("Fold: {}".format(i_fold+1))
    # new fold - network from scratch
    experiment = Experiment(device)
    model = Network(model_dict)
    params_dict["fold"] = i_fold+1
    # set the dataloaders for the fold
    train = torch.utils.data.Subset(dataset, train_index)
    test = torch.utils.data.Subset(dataset, test_index)
    train_loader = torch.utils.data.DataLoader(
        train, batch_size=params_dict["batch_size_train"], shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        test, batch_size=params_dict["batch_size_test"], shuffle=True)
    # set up the experiment
    experiment.set_metadata(params_dict)
    experiment.set_network(model_dict)
    experiment.update_network(model)
    experiment.set_loaders(train_loader, test_loader)
    experiment.set_loss(torch.nn.CrossEntropyLoss())

    # training loop
    for idx, epoch in enumerate(range(params_dict["n_epochs"])):
        print("Epoch: {}".format(epoch))
        epoch_vals = experiment.train_epoch(epoch)
        logging.info(epoch_vals)
        logging.info(experiment.network)
        experiment.save_weights({
            'epoch': epoch,
            'state_dict': experiment.network.state_dict(),
            'train_acc': experiment.tacc,
            'val_acc': experiment.acc,
            'train_loss': experiment.trainLoss,
            'val_loss': experiment.testLoss,
            'optimizer': experiment.optimizer.state_dict(),
            'traint': experiment.traint,
            'traini': experiment.traini,
            'params': experiment.params_dict
        }, 'models/{}_{}.pth.tar'.format(uid, epoch,))


# In[ ]:




