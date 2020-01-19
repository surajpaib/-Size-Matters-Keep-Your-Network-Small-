from BaseClass import BaseClass
import copy
import torch
import matplotlib.pyplot as plt
import time
import numpy as np
from captum.attr import LayerConductance


class Pruning(BaseClass):
    def __init__(self, percentage):
        self.set_percentage(percentage)

    def set_model(self, model):
        self.prev_model = model
        self.new_model = copy.deepcopy(model)
        #print("Set Model for Pruning: \n")
        #self.print_model_structure(self.prev_model)
        self.total_neurons = 0
        for idx, p in enumerate(self.new_model.named_parameters()):
            if len(p[1].data.size()) == 1 and p[0].split(".")[0] != 'output_layer':
                self.total_neurons += p[1].data.size()[0]

        #print("Total Neurons in Network: ", self.total_neurons)


    def set_test_data(self, data):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.test_data = data[0].to(device)
        self.test_target = data[1].to(device)


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
        self.neurons_retained = []
        layer_importances = self.layer_importance(self.strategy.__name__)[:-1]
        layer_importances = np.array([1/v for v in layer_importances])
        total_neurons_to_prune = self.total_neurons * self.pruning_perc
        #print(total_neurons_to_prune)

        neurons_to_prune = list(layer_importances/ np.sum(layer_importances) * total_neurons_to_prune) + [0]

        neurons_to_prune = [val for val in neurons_to_prune for _ in (0, 1)]
        #print(neurons_to_prune)

        for idx, p in enumerate(self.new_model.named_parameters()):
            prune_idx = self.strategy(p, int(np.round(neurons_to_prune[idx])))
            if len(prune_idx) > 0:
                self.neurons_retained.append(prune_idx)


    def l1_norm_pruning(self, p, n_neurons):

        p = p[1]
        if len(p.data.size()) != 1:
            normed_weights = p.data.abs()
            l1_norm_layer = [torch.sum(normed_weights[neuron_idx, :]).item() for neuron_idx in range(normed_weights.shape[0])]
            try:
                prune_idx = np.argpartition(np.array(l1_norm_layer), -(p.data.size()[0] - n_neurons))
                prune_idx = prune_idx[-(p.data.size()[0] - n_neurons):]
            except:
                prune_idx = []
        else:
            prune_idx = []

        return prune_idx


    def layer_conductance_pruning(self, p, n_neurons):

        _layer_name = p[0].split(".")
        if len(_layer_name) == 3:
            layer_name = _layer_name[0] + '[' + _layer_name[1] + ']'
        elif len(_layer_name) == 2:
            layer_name = _layer_name[0]


        if len(p[1].data.size()) != 1:

            cond = LayerConductance(self.new_model, eval('self.new_model.' + layer_name))
            cond_vals = cond.attribute(self.test_data,target=self.test_target)
            cond_vals = np.abs(cond_vals.cpu().detach().numpy())
            neuron_values = np.mean(cond_vals, axis=0)
            # Do we really need visualization?
            # visualize_importances(cond_vals.shape[1], neuron_values, p[0] + '{}'.format(time.time()))
            try:
                prune_idx = np.argpartition(np.array(neuron_values), -(p[1].data.size()[0] - n_neurons))
                prune_idx = prune_idx[-(p[1].data.size()[0] - n_neurons):]
            except:
                prune_idx = []
            #print("Neurons Retained", len(prune_idx))
        else:
            prune_idx = []

        return prune_idx


    def layer_importance(self, strategy):
        layer_importance_list = []


        if strategy == "l1_norm_pruning":
            for p in self.new_model.parameters():
                if len(p.data.size()) != 1:
                    normed_weights = p.data.abs()
                    layer_importance_list.append(torch.mean(normed_weights).item())

            return layer_importance_list


        elif strategy == "layer_conductance_pruning":
            for p in self.new_model.named_parameters():

                _layer_name = p[0].split(".")
                if len(_layer_name) == 3:
                    layer_name = _layer_name[0] + '[' + _layer_name[1] + ']'
                elif len(_layer_name) == 2:
                    layer_name = _layer_name[0]

                if len(p[1].data.size()) != 1:
                    cond = LayerConductance(self.new_model, eval('self.new_model.' + layer_name))
                    cond_vals = cond.attribute(self.test_data,target=self.test_target)
                    cond_vals = np.abs(cond_vals.cpu().detach().numpy())

                    layer_importance_val = np.mean(cond_vals)
                    layer_importance_list.append(layer_importance_val)

            return layer_importance_list

    def apply_strategy(self):
        self.define_strategy()

        # Create a copy of the neurons retained for each hidden layer weights, for example: [400, 50] and add another copy for the biases [400]
        self.neurons_retained = [val for val in self.neurons_retained for _ in (0, 1)]

        # Convert parameters to list
        self.param_list = list(self.new_model.parameters())

        # Iterate over all copy of neurons.
        for (i, neuron_idx) in enumerate(self.neurons_retained):
            # Get all weights for a particular layer, including weights and biases.
            idx_weights = self.param_list[i]

            # Condition check for all layers except the last 2. Last two layers do not prune number of neurons.
            if i < len(self.param_list) - 2:
                # print(neuron_idx)

                # Set y as weights of all neurons to keep
                y = idx_weights[neuron_idx]

                # If the layer is not the first two layers (associated to input layers) and the layers are weights (weights have a shape of 2, biases have a shape of 1)
                if i > 1 and len(idx_weights.shape)> 1:
                    # Modify y in the second index to only keep the neurons in the previous layer.
                    y = y[:, self.neurons_retained[i-1]]

            # If the layer is not the first 2 layers and is a weights layer then set the second index to neurons retained from previous layer.This condition only works for last 2 layers.
            elif i > 1 and len(idx_weights.shape) > 1:
                y = idx_weights[:, self.neurons_retained[i-1]]

            # For all other layers set y as just all the weights.
            else:
                y = idx_weights


            # Set the weights data to y. editing the model parameters.
            idx_weights.data = y


    def get_model(self):
        #print("Get Model after Pruning: \n")
        #self.print_model_structure(self.new_model)
        return self.new_model

    def get_optimizer(self):
        return self.prev_optimizer

    def get_optimizer_model(self):
        return self.get_optimizer(), self.get_model()

    def prune_model(self, optimizer, model, strategy):
        self.strategy = strategy
        self.set_optimizer_model(optimizer, model)
        self.apply_strategy()
        return self.get_optimizer_model()



def visualize_importances(neuron_no, importances, name='1'):
    x_pos = (np.arange(neuron_no))
    plt.figure(figsize=(12,6))
    plt.bar(x_pos, importances, align='center')
    plt.savefig("{}.png".format(name))
