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


def get_layer_type_scheme(params):
    types_list = []

    for hidden_layer in params['hidden_layer']:
        if 'type' in hidden_layer:
            layer_type = hidden_layer['type']
            if layer_type in dir(nn):
                types_list.append(eval('nn.'+layer_type))
                continue
        types_list.append(nn.Linear)

    if 'type' in params['output_layer']:
        layer_type = params['output_layer']['type']
        if layer_type in dir(nn):
            types_list.append(eval('nn.'+layer_type))
        else:
            types_list.append(nn.Linear)
    else:
        types_list.append(nn.Linear)

    return types_list


def freq_dist(iterations, dist):
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
