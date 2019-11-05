from includes import *
from utils import *

class FullyConnectedNetwork(nn.Module):
    def __init__(self, params):
        super(FullyConnectedNetwork, self).__init__()

        # Validate Structure of Input
        if not(input_validation(params)):
            exit()

        network_params = params['network']
        # Construct the Network Definition
        self.input_layer = nn.Linear(network_params['input_layer']['units'], network_params['hidden_layer'][0]['units'])
        self.hidden_layers = []
        for i in range(1, len(network_params['hidden_layer'])):
            self.hidden_layers.append(nn.Linear(network_params['hidden_layer'][i-1]['units'], network_params['hidden_layer'][i]['units']))
        self.output_layer = nn.Linear(network_params['hidden_layer'][-1]['units'], network_params['output_layer']['units'])        
        
        
        # Get choice of activations or if missing default to values
        self.activation_scheme = get_activation_scheme(network_params)     
        print(self.activation_scheme)   

    def forward(self, x):
        print(self.activation_scheme)
        x = self.activation_scheme[0](self.input_layer(x))
        for (i, layer) in enumerate(self.hidden_layers):
            x = self.activation_scheme[i+1](layer(x))
        x = self.activation_scheme[-1](self.output_layer(x))
        return x




if __name__ == "__main__":
    # Provide Network Description Here. Scalable to any number of units and activations within the Pytorch Functional API
    nn = FullyConnectedNetwork({
        "network":{
        'input_layer': {
            "units": 784
            },
        'hidden_layer': [{
            "units": 100, 
            }, {
                "units": 4, 
                "activation": "relu"
                }, {
                    "units": 5, 
                    "activation": "relu"
                    },
                    {
                    "units": 5, 
                    "activation": "relu"
                    },
                    {
                    "units": 5, 
                    "activation": "relu"
                    }],
        'output_layer': {
            "units": 20,
            "activation": "softmax"
            }
        }
    })


    # Print Network Description and test network forward and backprop with dummy IO

    print(list(nn.parameters()))
    input = torch.randn(1, 1, 784)
    target = torch.randn(1, 1, 20)
    out = nn(input)
    logging.info("Output: {} \n".format(out))
    loss = torch.nn.functional.mse_loss(out, target)
    logging.info("Loss: {} \n".format(loss))
    loss.backward()

