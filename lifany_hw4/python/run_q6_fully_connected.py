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
max_iters = 2
learning_rate = 1e-3
lossf = nn.CrossEntropyLoss()
optimizer = optim.SGD(myNet.parameters(), lr=learning_rate)

# Training loop
for itr in range(max_iters):
    total_loss = 0.0
    total_correct = 0.0
    total_instances = 0

    for times, data in enumerate(trainLoader):
        inputs, labels = data[0].to(device), data[1].to(device)
        inputs = inputs.view(inputs.shape[0], -1)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Foward, backward, optimize
        outputs = myNet(inputs)
        loss = lossf(outputs, labels)
        loss.backward()
        optimizer.step()

        # Total loss
        total_loss += loss.item()

        # average accuracy
        classifications = torch.argmax(myNet(inputs), dim=1)
        correct_predictions = sum(classifications==labels).item()
        total_correct+=correct_predictions
        total_instances+=len(inputs)
    accuracy = round(total_correct/total_instances, 4)
    
    if itr % 2 == 0:
        print(
            "itr: {:02d} \t loss: {:.2f} \t acc : {:.2f}".format(
                itr, total_loss, accuracy
            )
        )

# save the weights
torch.save(myNet, 'q6_fully_connected.pth')

