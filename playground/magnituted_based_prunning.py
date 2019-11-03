# https://github.com/yangzhangalmo/pytorch-iris/blob/master/main.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class Net(nn.Module):
    # define nn
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(4, 2)
        #self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(2, 3)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, X):
        X = F.relu(self.fc1(X))
        #X = self.fc2(X)
        X = self.fc3(X)
        X = self.softmax(X)

        return X

# Train a NN 

# load IRIS dataset
dataset = pd.read_csv('dataset/iris.csv')

# transform species to numerics
dataset.loc[dataset.species == 'Iris-setosa', 'species'] = 0
dataset.loc[dataset.species == 'Iris-versicolor', 'species'] = 1
dataset.loc[dataset.species == 'Iris-virginica', 'species'] = 2

train_X, test_X, train_y, test_y = train_test_split(dataset[dataset.columns[0:4]].values,
                                                    dataset.species.values, test_size=0.8)

# wrap up with Variable in pytorch
train_X = Variable(torch.Tensor(train_X).float())
test_X = Variable(torch.Tensor(test_X).float())
train_y = Variable(torch.Tensor(train_y).long())
test_y = Variable(torch.Tensor(test_y).long())

net = Net()

criterion = nn.CrossEntropyLoss()  # cross entropy loss

optimizer = torch.optim.SGD(net.parameters(), lr=0.01)

for epoch in range(10):
    optimizer.zero_grad()
    out = net(train_X)
    loss = criterion(out, train_y)
    loss.backward()
    optimizer.step()
    #print(net.fc1.weight)
    #if epoch % 100 == 0:
    #    print('number of epoch', epoch, 'loss', loss.data[0])

predict_out = net(test_X)
_, predict_y = torch.max(predict_out, 1)
'''
print(net)
print(net.fc1.weight)

print(net.fc1.weight[0][2])
print(net.fc1.weight[0][2])


print(net.fc3.weight)
'''


# Magnitude based prunning
pr = 1000
x = 0
y = 0

print()
print("###########")
print()

# Calculate Magnitude and prune it
for i in range(len(net.fc1.weight)):
    for ii in range(len(net.fc1.weight[i])):
        if pr > abs(net.fc1.weight[i][ii]):
            x = i
            y = ii
            pr = abs(net.fc1.weight[i][ii])

print(net.fc1.weight)
print(x)
print(y)

