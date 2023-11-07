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

# Load original data
np.random.seed(42)

train_data = scipy.io.loadmat("../data/nist36_train.mat")
valid_data = scipy.io.loadmat("../data/nist36_valid.mat")
test_data = scipy.io.loadmat("../data/nist36_test.mat")

train_x, train_y = train_data["train_data"], train_data["train_labels"]
valid_x, valid_y = valid_data["valid_data"], valid_data["valid_labels"]
test_x, test_y = test_data["test_data"], test_data["test_labels"]

trainLoader = np2loader(train_x, train_y, batchsize=64)
validLoader = np2loader(valid_x, valid_y, shuffling=False)
testLoader = np2loader(test_x, test_y, shuffling=False)

# Call the network
myNet = Net()

# Parameters
max_iters = 200
learning_rate = 5e-2
lossf = nn.CrossEntropyLoss()
optimizer = optim.SGD(myNet.parameters(), lr=learning_rate)
fname = 'q6_fully_connected.pth'

# Training loop
training_loop(myNet, trainLoader, validLoader, device, max_iters, learning_rate, lossf, optimizer, fname)

# Test
myNet.load_state_dict(torch.load(fname))
test_acc, _ = evaluate_model(myNet, testLoader, lossf, device)
print("Test accuracy: ", test_acc)
