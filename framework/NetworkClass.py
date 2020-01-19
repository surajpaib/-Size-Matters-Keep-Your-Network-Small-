from includes import *
from pgs_utils import get_activation_scheme, input_validation, get_layer_type_scheme


class Network(nn.Module):
    """
    """

    def __init__(self, params):
        super(Network, self).__init__()

        # Validate Structure of Input
        if not input_validation(params):
            exit()

        network_params = params['network']
        self.type_scheme = get_layer_type_scheme(network_params)

        # Construct the Network Definition
        self.input_layer = self.type_scheme[0](network_params['input_layer']['units'],
                                               network_params['hidden_layer'][0]['units'])

        # equal random weights
        weight_init_bounds = 0.03
        torch.nn.init.uniform_(self.input_layer.weight, -
                               weight_init_bounds, weight_init_bounds)

        self.hidden_layers = nn.ModuleList()
        for i in range(len(network_params['hidden_layer']) - 1):
            self.hidden_layers.append(self.type_scheme[i](network_params['hidden_layer'][i]['units'],
                                                          network_params['hidden_layer'][i+1]['units']))

            # equal random weights
            torch.nn.init.uniform_(
                self.hidden_layers[-1].weight, -weight_init_bounds, weight_init_bounds)

        self.output_layer = self.type_scheme[-1](network_params['hidden_layer'][-1]['units'],
                                                 network_params['output_layer']['units'])
        # Get choice of activations or if missing default to values
        self.activation_scheme = get_activation_scheme(network_params)

    def forward(self, x):
        x = self.activation_scheme[0](self.input_layer(x))
        for (i, layer) in enumerate(self.hidden_layers):
            x = self.activation_scheme[i+1](layer(x))
        x = self.activation_scheme[-1](self.output_layer(x))
        return x

    def set_optimizer(self, optim):
        pass

    def set_loss(self, loss):
        pass


if __name__ == "__main__":
    # Provide Network Description Here.
    # Scalable to any number of units and activations within the Pytorch Functional API
    nn = Network({
        "network": {
            'input_layer': {
                "units": 784,

            },
            'hidden_layer': [{
                "units": 100,
                "type": "Linear"
            },
                {
                    "units": 4,
                    "activation": "relu",
                    "type": "Linear"

            },
                {
                    "units": 5,
                    "activation": "relu",
                    "type": "Linear"

            },
                {
                    "units": 5,
                    "activation": "relu",
                    "type": "Linear"

            },
                {
                    "units": 5,
                    "activation": "relu",
                    "type": "Linear"

            }],
            'output_layer': {
                "units": 20,
                "activation": "softmax",
                "type": "Linear"

            }
        }
    })

    # Print Network Description and test network forward and backprop with dummy IO

    input = torch.randn(1, 1, 784)
    target = torch.randn(1, 1, 20)
    out = nn(input)
    logging.info("Output: {} \n".format(out))
    loss = torch.nn.functional.mse_loss(out, target)
    logging.info("Loss: {} \n".format(loss))
    loss.backward()
