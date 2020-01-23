from BaseClass import BaseClass
import copy
import torch
import torch.nn as nn
import numpy as np
from captum.attr import LayerConductance

# import slack


# def send_slack_message(message):

#     slack_token = 'TOKEN HERE'
#     client = slack.WebClient(token=slack_token)
#     response = client.chat_postMessage(
#         channel='bots',
#         text=message)


class Growing(BaseClass):
    def __init__(self, percentage):
        self.set_percentage(percentage)

    def set_model(self, model):
        self.prev_model = model
        self.new_model = copy.deepcopy(model)
        #print("Set Model for Growing: \n")
        # self.print_model_structure(self.prev_model)

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

    def set_percentage(self, growing_perc):
        self.growing_perc = growing_perc

    def print_model_structure(self, model):
        for (layer, param) in enumerate(model.parameters()):
            print("Layer {} , Parameters: {}".format(layer, param.shape))

    def random_growing(self):

        # Number of numbers to add per layer, list of integres
        self.number_neurons_per_layer = []

        # Current strategy: add random number of neurons between 0 and 9.
        for p in self.new_model.parameters():
            self.number_neurons_per_layer.append(np.random.randint(10))

    def l1_norm_growing(self):

        # Number of numbers to add per layer, list of integres
        self.number_neurons_per_layer = []

        meaned_l1_layer = []
        neurons_per_layer = []

        for p in [x for x in self.new_model.parameters()][:-2]:  # skip output layer

            if len(p.data.size()) != 1:  # skip biases
                neurons_per_layer.append(p.data.shape[0])
                normed_weights = p.data.abs()
                l1_norm_layer = torch.mean(normed_weights).item()
                meaned_l1_layer.append(l1_norm_layer)

        # dont take into account output layer
        total_number_neurons = np.nansum(neurons_per_layer)
        total_l1 = np.nansum(meaned_l1_layer)

        #add_per_layer = [int(round((x/total_l1)*total_number_neurons*self.growing_perc,0)) for x in meaned_l1_layer]

        add_per_layer = []

        for x in meaned_l1_layer:
            try:
                add_p_layer = int(
                    round((x/total_l1)*total_number_neurons*self.growing_perc, 0))
            except:
                print('Could not calculate layer cond value, so 0 used, parameters:')
                print('total_cond ', total_l1)
                print('total number neurons ', total_number_neurons)
                print('cond of layer ', x)
                add_p_layer = 0
                # send_slack_message('growing except has occured')

            add_per_layer.append(add_p_layer)

        self.number_neurons_per_layer = add_per_layer

    def layer_conductance_growing(self):

        # Number of numbers to add per layer, list of integres
        self.number_neurons_per_layer = []

        meaned_cond_layer = []
        neurons_per_layer = []

        for p in [x for x in self.new_model.named_parameters()][:-2]:  # skip output layer

            if len(p[1].data.size()) != 1:  # skip biases

                _layer_name = p[0].split(".")
                if len(_layer_name) == 3:
                    layer_name = _layer_name[0] + '[' + _layer_name[1] + ']'
                elif len(_layer_name) == 2:
                    layer_name = _layer_name[0]

                neurons_per_layer.append(p[1].data.shape[0])

                cond = LayerConductance(self.new_model, eval(
                    'self.new_model.' + layer_name))
                cond_vals = cond.attribute(
                    self.test_data, target=self.test_target)
                cond_vals = cond_vals.cpu().detach().numpy()
                layer_value = np.nanmean(np.absolute(cond_vals))
                meaned_cond_layer.append(layer_value)

        # dont take into account output layer
        total_number_neurons = np.nansum(neurons_per_layer)
        total_cond = np.nansum(meaned_cond_layer)

        add_per_layer = []

        for x in meaned_cond_layer:
            try:
                add_p_layer = int(
                    round((x/total_cond)*total_number_neurons*self.growing_perc, 0))
            except:
                print('Could not calculate layer cond value, so 0 used, parameters:')
                print('total_cond ', total_cond)
                print('total number neurons ', total_number_neurons)
                print('cond of layer ', x)
                add_p_layer = 0
                # send_slack_message('growing except has occured')

            add_per_layer.append(add_p_layer)

        self.number_neurons_per_layer = add_per_layer

    def define_strategy(self):
        self.strategy()

    def apply_strategy(self):
        self.define_strategy()

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Double the list because it passes for both weights and bias
        # list with number of neurons to add per layer

        self.number_neurons_per_layer = [
            val for val in self.number_neurons_per_layer for _ in (0, 1)]

        # layers
        self.param_list = list(self.new_model.parameters())

        for (i, number_new_neurons) in enumerate(self.number_neurons_per_layer):

            # note: every uneven is weights / every uneven is bias
            # uneven is bias, skip this
            if i % 2 != 0:
                continue

            # dont add neurons to output layer
            if i >= len(self.param_list) - 2:
                break

            # get layers' weights and biases
            layer_weights_1 = self.param_list[i]
            layer_bias_1 = self.param_list[i+1]
            layer_weights_2 = self.param_list[i+2]

            # get number of neurons
            current_num_nodes_1 = layer_weights_1.shape[1]
            current_num_nodes_2 = layer_weights_2.shape[0]

            # Create new tensors
            add_weights_1 = torch.zeros(
                [number_new_neurons, current_num_nodes_1])
            add_bias_1 = torch.zeros([1, number_new_neurons])
            add_weights_2 = torch.zeros(
                [current_num_nodes_2, number_new_neurons])

            add_weights_1 = add_weights_1.to(device)
            add_bias_1 = add_bias_1.to(device)
            add_weights_2 = add_weights_2.to(device)

            print("-"*50)
            print((torch.flatten(layer_weights_1.data)).std())
            print("-"*50)

            # Randomize
            nn.init.normal_(add_weights_1, mean=layer_weights_1.data.mean(
            ).item(), std=(torch.flatten(layer_weights_1.data)).std())
            nn.init.normal_(add_bias_1, mean=layer_bias_1.data.mean(
            ).item(), std=(torch.flatten(layer_bias_1.data)).std())
            nn.init.normal_(add_weights_2, mean=layer_weights_2.data.mean(
            ).item(), std=(torch.flatten(layer_weights_2.data)).std())

            # merge weights
            new_weights_1 = torch.cat(
                [layer_weights_1, add_weights_1], dim=0)  # add bottom row
            new_bias_1 = torch.cat(
                [layer_bias_1, add_bias_1[0]])  # add bottom row
            new_weights_2 = torch.cat(
                [add_weights_2, layer_weights_2], dim=1)  # add first column

            # update weights
            layer_weights_1.data = new_weights_1
            layer_weights_2.data = new_weights_2
            layer_bias_1.data = new_bias_1

    # copy pasta from pruning
    def get_model(self):
        #print("Get Model after Growing: \n")
        # self.print_model_structure(self.new_model)
        return self.new_model

    def get_optimizer(self):
        return self.prev_optimizer

    def get_optimizer_model(self):
        return self.get_optimizer(), self.get_model()

    def grow_model(self, optimizer, model, strategy):
        self.strategy = strategy
        self.set_optimizer_model(optimizer, model)
        self.apply_strategy()
        return self.get_optimizer_model()
