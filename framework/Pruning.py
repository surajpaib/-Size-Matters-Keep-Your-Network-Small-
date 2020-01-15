from BaseClass import BaseClass
import copy
import torch
import numpy as np

class Pruning(BaseClass):
    def __init__(self, percentage):
        self.set_percentage(percentage)

    def set_model(self, model):
        self.prev_model = model
        self.new_model = copy.deepcopy(model)
        print("Set Model for Pruning: \n")
        self.print_model_structure(self.prev_model)


    def set_optimizer(self, optim):
        self.prev_optimizer = optim
        self.lr = self.prev_optimizer.state_dict()['param_groups'][0]['lr']

    def set_optimizer_model(self, optim, model):
        self.set_model(model)
        self.set_optimizer(optim)

    def set_percentage(self, pruning_perc):
        self.pruning_perc = pruning_perc

    def print_model_structure(self, model):
        for (layer, param) in enumerate(model.parameters()):
            print("Layer {} , Parameters: {}".format(layer, param.shape))


    def define_strategy(self):
        """
            Neurons retained should contain a list of neurons to be kept at each layer with hidden units.
        """

        # Access model weights
        self.neurons_retained = []

        for p in self.new_model.parameters():
                if len(p.data.size()) != 1:
                    normed_weights = p.data.abs()
                    l1_norm_layer = [torch.sum(normed_weights[neuron_idx, :]).item() for neuron_idx in range(normed_weights.shape[0])]
                    threshold = np.percentile(np.array(l1_norm_layer), self.pruning_perc * 100)
                    prune_idx = np.argwhere(np.array(l1_norm_layer) > threshold).flatten()
                    self.neurons_retained.append(prune_idx)


    def apply_strategy(self):
        self.define_strategy()
        self.neurons_retained = [val for val in self.neurons_retained for _ in (0, 1)]

        self.param_list = list(self.new_model.parameters())

        for (i, neuron_idx) in enumerate(self.neurons_retained):
            idx_weights = self.param_list[i]
            if i < len(self.param_list) - 2:
                print(neuron_idx)
                y = idx_weights[neuron_idx]
                if i > 1 and len(idx_weights.shape)> 1:
                    y = y[:, self.neurons_retained[i-1]]
            elif i > 1 and len(idx_weights.shape) > 1:
                y = idx_weights[:, self.neurons_retained[i-1]]

            else:
                y = idx_weights

            idx_weights.data = y


    def get_model(self):
        print("Get Model after Pruning: \n")
        self.print_model_structure(self.new_model)
        return self.new_model

    def get_optimizer(self):
        self.new_optimizer = torch.optim.SGD(self.new_model.parameters(), lr=self.lr)
        self.new_optimizer.load_state_dict(self.prev_optimizer.state_dict())
        return self.new_optimizer

    def get_optimizer_model(self):
        return self.get_optimizer(), self.get_model()

    def prune_model(self, optimizer, model):
        self.set_optimizer_model(optimizer, model)
        self.apply_strategy()
        return self.get_optimizer_model()
