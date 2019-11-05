from includes import *

def input_validation(params):
    if not('network' in params):
        logging.error("Network Definition: network key missing")
        return False

    network_params = params['network']

    if not({'input_layer', 'hidden_layer', 'output_layer'} <= set(network_params)):
        logging.error("Network Definition Incomplete")
        return False

    if len(network_params['hidden_layer']) < 1:
        logging.error("Not Enough Hidden Layers in the Definition")
        return False


    for param in network_params.keys(): 
        if (type(network_params[param]) == dict) and not('units' in network_params[param]):
            logging.error('Units Not Specified in the Input or Output Layers')
            return False
        if(type(network_params[param] == list)):
            for subparam in network_params[param]:
                if (type(subparam) == dict) and not('units' in subparam):
                    logging.error('Units Not Specified in the Hidden Layers')
                    return False

    return True


def get_activation_scheme(params):
    activations_list = []

    for hidden_layer in params['hidden_layer']:
        if 'activation' in hidden_layer:
            activation_function = hidden_layer['activation']
            if activation_function in dir(F):
                activations_list.append(eval('F.'+activation_function))
                continue
        activations_list.append(F.relu)
        
    if 'activation' in params['output_layer']:
        activation_function = params['output_layer']['activation']
        print(activation_function)
        if activation_function in dir(F):
            activations_list.append(eval('F.'+activation_function))
        else:
            activations_list.append(F.softmax)
    else:
        activations_list.append(F.softmax)

    return activations_list