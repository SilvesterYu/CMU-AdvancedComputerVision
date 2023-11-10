import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import numpy as np
import scipy
from nnq6 import *

# GPU or CPU
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print('GPU State:', device)

################################## Q6.1.2 #######################################
# Call the network
myCNN = CNNcifar()

# Parameters
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# Parameters
max_iters = 50
learning_rate = 5e-3
lossf = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(myCNN.parameters(), lr=learning_rate)
fname = 'q6_cifar_cnn.pth'
batch_size = 64

# Dataloaders
trainset = torchvision.datasets.CIFAR10(root='../data', train=True,
                                        download=True, transform=transform)
trainLoaderCNN = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='../data', train=False,
                                       download=True, transform=transform)
testLoaderCNN = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Training loop, comment this line when only doing testing
training_loop(myCNN, trainLoaderCNN, testLoaderCNN, device, max_iters, learning_rate, lossf, optimizer, fname, False)

# Test
myCNN.load_state_dict(torch.load(fname))
test_acc, _ = evaluate_model(myCNN, testLoaderCNN, lossf, device, False)
print("Test accuracy: ", test_acc)