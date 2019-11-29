import os

for epochs in [20, 25, 50]:
    for prune_perc in [0.05, 0.1, 0.2]:
        for prune_iter in [4, 5, 10]:
            print("*************************************************************************************")
            print("Epochs: {}  Pruning Percentage: {} Pruning Iterations: {}".format(epochs, prune_perc, prune_iter))
            os.system("python3 playground/L1_norm_based_pruning.py --epochs {} --pruning_perc {} --prune_iter {}".format(epochs, prune_perc, prune_iter))

