import sys
import logging
import time
import shutil

import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import KFold

sys.path.append('framework')
from NetworkClass import Network
from Pruning import Pruning
from Growing import Growing
from train_utils import ReshapeTransform

logging.basicConfig(level=logging.INFO)

class Experiment:
    def __init__(self, device):
        self.is_best = True
        self.device = device
        self.bestLoss = 999999
        self.tensorboard_summary = SummaryWriter()


    def set_network(self, network):
        self.network = network
        self.network = self.network.to(self.device)

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
        self.tensorboard_summary.add_scalar('Loss/train', loss_dict['train'], loss_dict['epoch'])
        self.tensorboard_summary.add_scalar('Loss/val', loss_dict['val'], loss_dict['epoch'])
        self.tensorboard_summary.add_scalar('Accuracy/train', loss_dict['tacc'], loss_dict['epoch'])
        self.tensorboard_summary.add_scalar('Accuracy/val', loss_dict['acc'], loss_dict['epoch'])
        self.tensorboard_summary.add_scalar('Time/train', loss_dict['traint'], loss_dict['epoch'])
        self.tensorboard_summary.add_scalar('Time/inference', loss_dict['traini'], loss_dict['epoch'])
        self.tensorboard_summary.add_text('params', str(self.params_dict), loss_dict['epoch'])

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

        if iterations == 50:
            exp_factor = 1.08005
            const_factor = 1
        elif iterations == 25:
            exp_factor = 1.17755
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
        trainLoss = 0.0
        correct = 0.0
        self.network.train()
        start_t = time.time()
        for batch_idx, (data, target) in enumerate(self.trainLoader):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.network(data)
            loss = self.loss(output, target)
            logging.info("Batch : {} \t Loss: {}".format(batch_idx, loss.item()))
            loss.backward()
            trainLoss += loss.item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
            self.optimizer.step()

        tacc = 100. * correct / len(self.trainLoader.dataset)
        traint = time.time() - start_t
        trainLoss = trainLoss * self.trainLoader.batch_size / len(self.trainLoader.dataset)



        testLoss = 0.0
        correct = 0.0
        self.network.eval()
        start_i = time.time()
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.testLoader):
                data, target = data.to(self.device), target.to(self.device)
                output = self.network(data)
                loss = self.loss(output, target)
                testLoss += loss.item()
                pred = output.data.max(1, keepdim=True)[1]
                correct += pred.eq(target.data.view_as(pred)).sum()
            acc = 100. * correct / len(self.testLoader.dataset)
            traini = (time.time() - start_i)/len(self.testLoader.dataset)
            testLoss = testLoss * self.testLoader.batch_size /len(self.testLoader.dataset)

        logging.info("VALIDATION: \t Loss: {}, Accuracy : {}".format(testLoss, acc))
        self.save_tensorboard_summary({'train':trainLoss, 'val': testLoss, 'acc': acc, 'epoch': epoch, 'traint': traint, 'traini': traini, 'tacc':tacc})

        self.bestLoss = min(testLoss, self.bestLoss)
        self.is_best = (self.bestLoss == testLoss)

        self.save_weights({
                'epoch': epoch,
                'state_dict': self.network.state_dict(),
                'best_acc1': self.bestLoss,
                'optimizer' : self.optimizer.state_dict(),
                'traint': traint,
                'traini': traini,
                'params': self.params_dict
            })



if __name__ == "__main__":
    params_dict = {
        "batch_size_train": 100,
        "learning_rate": 0.01,
        "batch_size_test": 1000,
        "n_epochs": 100,
        "type": "Growing",
        "percentage": 0.0582,
        "iterations": 50,
        "distribution": "equal"
    }
    # batch_size_train = 100
    # learning_rate = 0.01
    seed = 42
    # batch_size_test = 1000
    # n_epochs = 100

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    else:
        torch.manual_seed(seed)


    experiment = Experiment(device)

    if params_dict["type"] == "Pruning":
        pruning = Pruning(percentage=params_dict["percentage"])
    elif params_dict["type"] == "Growing":
        growing = Growing(percentage=params_dict["percentage"])


    model_dict = {
        "network":{
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

                }],
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
        ReshapeTransform((-1,))
        ]))

    test_dataset  = torchvision.datasets.FashionMNIST('../data/', train=False, download=True,
        transform=torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        ReshapeTransform((-1,))
        ]))

    dataset = torch.utils.data.ConcatDataset([train_dataset, test_dataset])

    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    for i_fold, (train_index, test_index) in enumerate(kf.split(dataset)):
        print("FOLD: {}".format(i_fold+1))
        # new fold - network from scratch
        model = Network(model_dict)
        params_dict["fold"] = i_fold+1
        # set the dataloaders for the fold
        train = torch.utils.data.Subset(dataset, train_index)
        test = torch.utils.data.Subset(dataset, test_index)
        train_loader = torch.utils.data.DataLoader(train, batch_size=params_dict["batch_size_train"], shuffle=True)
        test_loader = torch.utils.data.DataLoader(test, batch_size=params_dict["batch_size_test"], shuffle=True)

        # set up the experiment
        experiment.set_network(model)
        experiment.set_loaders(train_loader, test_loader)
        experiment.set_loss(torch.nn.CrossEntropyLoss())
        experiment.set_optimizer(torch.optim.SGD(model.parameters(), lr=params_dict["learning_rate"]))
        experiment.set_metadata(params_dict)
        iter_list = experiment.get_iteration_distribution(params_dict["iterations"], params_dict["distribution"])
        # training loop
        for idx, epoch in enumerate(range(params_dict["n_epochs"])):
            if iter_list[idx]:
                growing.set_test_data(next(iter(test_loader)))
                optimizer, model = growing.grow_model(experiment.optimizer, experiment.network, growing.l1_norm_growing)
                experiment.set_optimizer(optimizer)
                experiment.set_network(model)

            experiment.train_epoch(epoch)
