# Framework

This folder will contain the code for the framework. Current agreement on the framework's structure is:

    .
    ├── BaseClass.py                 # Base for the classes below (parent class)
    ├── Pruning.py                   # Incorporates various pruning strategies
    ├── Growing.py                   # Incorporates various growing strategies
    ├── Shifting.py		     # Combines the previous two to achieve shifting 
    └── README.md
   
   
   
## Some ideas from the meeting
### Config file
Yet to be seen.

### A network is encapsulated in Pruning/Growing/Shifting class

	network = torchvision.models.resnet18()
    # an idea 
    network = ourframework.Pruning(network, strategy='weight_magnitude')
    
### Framework and Pytorch training loop
	# Defining and encapsulating the network
	network = torchvision.models.resnet18()
    network = ourframework.Pruning(network, iterations=10, strategy='weight_magnitude')
    
    # Training loop
    #  we do a number of iterations of network pruning and for each one of them separate training
    for iter in network.iterations: 
    	# the regular pytorch training for loop
        for e in range(epochs):
        	for input, labels in trainloader:
            	.
                .
                .
        # when the training for the current network is done, 
        # get the pruned network and do the training for it
        network = network.next_model() # pruned model
        
        # get the lr and other stuff from the current optimizer to the new one
        optimizer_new = optim.Adam(network.parameters())
        optimizer_new.load_state_dict(optimizer.state_dict())
        optimizer = optimizer_new
        
