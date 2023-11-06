import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as dset
from torchvision import datasets, transforms
import torch.optim as optim
import numpy as np
import scipy
import matplotlib.pyplot as plt

from nnq6 import *

# GPU or CPU
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print('GPU State:', device)

# Transform to Tensor type and normalize
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5,), (0.5,)),]
)

# Dataloaders
trainSet = datasets.MNIST(root='MNIST', download=True, train=True, transform=transform)
testSet = datasets.MNIST(root='MNIST', download=True, train=False, transform=transform)
trainLoader = dset.DataLoader(trainSet, batch_size=64, shuffle=True)
testLoader = dset.DataLoader(testSet, batch_size=64, shuffle=False)

myNet = Net().to(device)
print(myNet)

# Parameters
epochs = 200
learning_rate = 1e-3
lossf = nn.CrossEntropyLoss()
optimizer = optim.SGD(myNet.parameters(), lr=learning_rate)

