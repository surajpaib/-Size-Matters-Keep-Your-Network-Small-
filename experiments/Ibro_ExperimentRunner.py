import sys
import logging
import time
import shutil
import string
import random
import os

import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import KFold

sys.path.append('framework')
sys.path.append('application')

from NetworkClass import Network
from Experiment import Experiment
from Pruning import Pruning
from Growing import Growing
from train_utils import ReshapeTransform


def randomString(stringLength=10):
    """Generate a random string of fixed length """
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(stringLength))


if __name__ == "__main__":
    logging.basicConfig(level=logging.CRITICAL)

    if not(os.path.isdir('models')):
        os.mkdir('models')

    for distributions in ["equal", "incr", "decr"]:
        for perc_iter_tuple in [(0.1088*2, 0.1221, 20), (0.2057*2, 0.259, 10), (0.0718, 0.1547, 10), (0.0353, 0.0732, 20), (0.2057, 0.259, 10), (0.0353, 0.0367, 20)]:
            # for method in ["l1_norm", "layer_conductance"]:

            params_dict = {
                "batch_size_train": 100,
                "learning_rate": 0.01,
                "batch_size_test": 1000,
                "n_epochs": 100,
                "type": "Shifting",  # Pruning / Growing / Shifting
                "method": "layer_conductance",  # layer_conductance / l1_norm
                "method2": "layer_conductance",  # none / layer_conductance / l1_norm
                "percentage": perc_iter_tuple[0],
                "percentage2": perc_iter_tuple[1],
                "iterations": perc_iter_tuple[2],
                "distribution": distributions  # "equal", "incr", "decr"
            }
            # batch_size_train = 100
            # learning_rate = 0.01
            seed = 42
            uid = randomString(stringLength=6)
            # batch_size_test = 1000
            # n_epochs = 100

            device = torch.device(
                "cuda:0" if torch.cuda.is_available() else "cpu")
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
            else:
                torch.manual_seed(seed)

            experiment = Experiment(device)

            if params_dict["type"] == "Pruning":
                pruning = Pruning(percentage=params_dict["percentage"])
            elif params_dict["type"] == "Growing":
                growing = Growing(percentage=params_dict["percentage"])
            elif params_dict["type"] == "Shifting":
                growing = Growing(percentage=params_dict["percentage2"])
                pruning = Pruning(percentage=params_dict["percentage"])

            model_dict = {
                "network": {
                    'input_layer': {
                        "units": 784,

                    },
                    'hidden_layer': [{
                        "units": 168,
                        "activation": "relu",
                        "type": "Linear"
                    },
                        {
                            "units": 168,
                            "activation": "relu",
                            "type": "Linear"

                    },
                        {
                            "units": 168,
                            "activation": "relu",
                            "type": "Linear"

                    }
                    ],
                    'output_layer': {
                        "units": 10,
                        "activation": "softmax",
                        "type": "Linear"
                    }
                }
            }

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
                # new fold - network from scratch
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
                experiment.set_network(model)
                experiment.set_loaders(train_loader, test_loader)
                experiment.set_loss(torch.nn.CrossEntropyLoss())
                experiment.set_optimizer(torch.optim.SGD(
                    model.parameters(), lr=params_dict["learning_rate"]))
                experiment.set_metadata(params_dict)
                iter_list = experiment.get_iteration_distribution(
                    params_dict["iterations"], params_dict["distribution"])
                # training loop
                for idx, epoch in enumerate(range(params_dict["n_epochs"])):
                    if iter_list[idx]:

                        _type = params_dict['type'].lower()
                        _type_stam = _type[:-3]
                        _method_add = _type_stam + 'ing'
                        _method = params_dict['method']+'_'+_method_add

                        if _type_stam[-1] == 'n':
                            _type_stam += 'e'

                        if _type == 'shifting':
                            _type = 'pruning'
                            _type_stam = 'prune'
                            _method = params_dict['method']+'_pruning'

                        eval(_type).set_test_data(next(iter(test_loader)))
                        optimizer, model = eval(_type+'.'+_type_stam+'_model')(
                            experiment.optimizer, experiment.network, eval(_type+'.'+_method))
                        experiment.set_optimizer(optimizer)
                        experiment.set_network(model)

                        if params_dict['type'] == "Shifting" and params_dict['method2'] != "none":
                            # Growing after the pruning
                            _method2 = params_dict['method2']+'_growing'
                            growing.set_test_data(next(iter(test_loader)))
                            optimizer, model = growing.grow_model(
                                experiment.optimizer, experiment.network, eval('growing.'+_method2))
                            experiment.set_optimizer(optimizer)
                            experiment.set_network(model)

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
                        }, 'models/{}_{}_{}_{}.pth.tar'.format(uid, i_fold+1, epoch, _method))
                    print('Distribution: ', distributions, ' Percentage: ', str(
                        perc_iter_tuple), ' Fold ', str(i_fold), ' epoch ', str(epoch))
                    experiment.train_epoch(epoch)
