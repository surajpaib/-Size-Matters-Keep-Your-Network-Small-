import sys
import logging
import time
import shutil
import string
import random
import copy
import os

import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import KFold

sys.path.append('framework')

from NetworkClass import Network
from Pruning import Pruning
from Growing import Growing
from train_utils import ReshapeTransform
import numpy as np

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class Experiment:
    def __init__(self, device):
        self.is_best = True
        self.device = device
        self.bestLoss = 999999
        self.tensorboard_summary = SummaryWriter()


    def update_network(self, network):
        new_model_dict = copy.deepcopy(self.model_dict)
    
        for i, layer in enumerate(self.model_dict['network']['hidden_layer']):
            if i == 0:

                new_model_dict['network']['hidden_layer'][i]['units'] = network.state_dict()['input_layer.weight'].shape[0]
            else:
                new_model_dict['network']['hidden_layer'][i]['units'] = network.state_dict()['hidden_layers.{}.weight'.format(i-1)].shape[0]

        # Create new model based on updated network definition
        updated_model = Network(new_model_dict)
        updated_model.load_state_dict(network.state_dict())

        self.network = updated_model
        self.network = self.network.to(self.device)

    def set_network(self, model_dict):
        self.model_dict = model_dict
        network = Network(model_dict)
        self.network = network
        self.network = self.network.to(self.device)
        self.set_optimizer(torch.optim.SGD(self.network.parameters(), lr=self.params_dict["learning_rate"]))

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    def set_loaders(self, trainLoader, testLoader):
        self.trainLoader = trainLoader
        self.testLoader = testLoader

    def set_loss(self, loss):
        self.loss = loss

    def set_metadata(self, params_dict):
        self.params_dict = params_dict

    def save_weights(self, state, filename='./checkpoint.pth.tar'):
        torch.save(state, filename)
        if self.is_best:
            shutil.copyfile(filename, './model_best.pth.tar')

    def save_tensorboard_summary(self, loss_dict):
        self.tensorboard_summary.add_scalar(
            'Loss/train', loss_dict['train'], loss_dict['epoch'])
        self.tensorboard_summary.add_scalar(
            'Loss/val', loss_dict['val'], loss_dict['epoch'])
        self.tensorboard_summary.add_scalar(
            'Accuracy/train', loss_dict['tacc'], loss_dict['epoch'])
        self.tensorboard_summary.add_scalar(
            'Accuracy/val', loss_dict['acc'], loss_dict['epoch'])
        self.tensorboard_summary.add_scalar(
            'Time/train', loss_dict['traint'], loss_dict['epoch'])
        self.tensorboard_summary.add_scalar(
            'Time/inference', loss_dict['traini'], loss_dict['epoch'])
        self.tensorboard_summary.add_text(
            'params', str(self.params_dict), loss_dict['epoch'])

    def get_iteration_distribution(self, iterations, dist):
        """
        Dummy function for iteration distribution
        """
        dists = ["equal", "incr", "decr"]
        freq_lst = []
        epochs = 100
        if dist == dists[0]:
            final_lst = [False] + \
                ([True]+[False]*int((epochs/iterations)-1))*int(iterations)
            final_lst.pop()
            return(final_lst)

        if iterations == 20:
            exp_factor = 1.23397
            const_factor = 1.5
        elif iterations == 10:
            exp_factor = 1.52268
            const_factor = 3

        for i in range(1, iterations+1):
            increment = int(round((exp_factor**i) + i*const_factor))
            freq_lst.append(increment)

        comp_lst = range(1, epochs+1)
        final_lst = []

        for i in range(len(comp_lst)):
            if comp_lst[i] in freq_lst:
                final_lst.append(1)
            else:
                final_lst.append(0)

        if dist == dists[2]:
            final_lst = reversed(final_lst)

        return(list(map(bool, final_lst)))

    def train_epoch(self, epoch):
        self.trainLoss = 0.0
        correct = 0.0
        self.network.train()
        start_t = time.time()
        for batch_idx, (data, target) in enumerate(self.trainLoader):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.network(data)
            loss = self.loss(output, target)
            # logging.info("Batch : {} \t Loss: {}".format(batch_idx, loss.item()))
            loss.backward()
            self.trainLoss += loss.item()
            output = torch.argmax(output, dim=1)
            # print(output)
            # print(target)

            correct += (output == target).float().sum()

            # pred = output.data.max(1, keepdim=True)[1]
            # correct += pred.eq(target.data.view_as(pred)).sum()
            self.optimizer.step()

        self.tacc = 100. * correct / len(self.trainLoader.dataset)
     
        self.traint = time.time() - start_t
        self.trainLoss = self.trainLoss * \
            self.trainLoader.batch_size / len(self.trainLoader.dataset)

        self.testLoss = 0.0
        correct = 0.0
        self.network.eval()
        start_i = time.time()
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.testLoader):
                data, target = data.to(self.device), target.to(self.device)
                output = self.network(data)
                loss = self.loss(output, target)
                self.testLoss += loss.item()
                output = torch.argmax(output, dim=1)
                # print(output)
                # print(target)

                correct += (output == target).float().sum()
                # pred = output.data.max(1, keepdim=True)[1]
                # correct += pred.eq(target.data.view_as(pred)).sum()
            self.acc = 100. * correct / len(self.testLoader.dataset)
      

            self.traini = (time.time() - start_i)/len(self.testLoader.dataset)
            self.testLoss = self.testLoss * \
                self.testLoader.batch_size / len(self.testLoader.dataset)

        # logging.info("VALIDATION: \t Loss: {}, Accuracy : {}".format(testLoss, acc))
        self.save_tensorboard_summary({'train': self.trainLoss, 'val': self.testLoss, 'acc': self.acc,
                                       'epoch': epoch, 'traint': self.traint, 'traini': self.traini, 'tacc': self.tacc})

        self.bestLoss = min(self.testLoss, self.bestLoss)
        self.is_best = (self.bestLoss == self.testLoss)


        return {'training_time': self.traint, 'inference_time': self.traini, 'model_parameters': count_parameters(self.network), 'training_acc': self.tacc, 'test_acc': self.acc}


