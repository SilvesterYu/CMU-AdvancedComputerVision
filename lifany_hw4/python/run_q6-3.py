import torch
import torchvision
from torchvision.models import resnet101
import numpy as np
import scipy
from nnq6 import *
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as T
import pickle

if torch.cuda.is_available():
    device = 'cuda:0'  
else:
    device = 'cpu'
print('GPU State:', device)

################################ Q 6.3 ############################
# Load the model
myNet = resnet101(pretrained=True)

# Parameters
data_dir = "../data/Imagenet32_val/"
max_iters = 100
learning_rate = 1e-3
batch_size = 64
numworkers = 2
lossf = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(myNet.parameters())
fname = "q6_imagenet_resnet.pth"

# Dataloaders
val_data = pickle.load(data_dir + "val_data")
print(val_data)

