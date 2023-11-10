import torch
import torchvision
from torchvision.models import squeezenet1_1, SqueezeNet1_1_Weights
import numpy as np
import scipy
from nnq6 import *
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision.transforms as T
from torchvision.datasets import ImageFolder

if torch.cuda.is_available():
    device = 'cuda:0'  
else:
    device = 'cpu'
print('GPU State:', device)

################################ Q 6.2 ############################
# Load the pretrained network squeezenet1_1
myNet = squeezenet1_1(SqueezeNet1_1_Weights)
print(myNet)

# Parameters
data_dir = "../data/oxford-flowers17/"
max_iters = 100
learning_rate = 1e-3
batch_size = 64
numworkers = 2
lossf = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(myNet.parameters())
fname = "q6_flowers.pth"

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

train_transform = T.Compose([
    T.Resize(224),
    T.RandomResizedCrop(224),
    T.RandomHorizontalFlip(),
    T.ToTensor(),            
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
  ])

val_transform = T.Compose([
    T.Resize(224),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
  ])

# Dataloaders
train_dset = ImageFolder(data_dir + "train", transform=train_transform)
train_loader = DataLoader(train_dset,
                    batch_size=batch_size,
                    num_workers=numworkers,
                    shuffle=True)

val_dset = ImageFolder(data_dir + "val", transform=val_transform)
val_loader = DataLoader(val_dset,
                    batch_size=batch_size,
                    shuffle=False, 
                num_workers=numworkers)

test_dset = ImageFolder(data_dir + "test", transform=val_transform)
test_loader = DataLoader(test_dset,
                    batch_size=batch_size,
                    shuffle=False, 
                num_workers=numworkers)

# Training loop
training_loop(myNet, train_loader, val_loader, device, max_iters, learning_rate, lossf, optimizer, fname, False)

# Test
myNet.load_state_dict(torch.load(fname))
test_acc, _ = evaluate_model(myNet, test_loader, lossf, device, False)
print("Test accuracy: ", test_acc)