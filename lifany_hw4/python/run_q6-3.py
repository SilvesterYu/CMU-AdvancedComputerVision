import torch
import torchvision
from torchvision.models import resnet50
import numpy as np
import scipy
from nnq6 import *
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset,DataLoader
import torchvision.transforms as T
import pickle

if torch.cuda.is_available():
    device = 'cuda:0'  
else:
    device = 'cpu'
print('GPU State:', device)

################################ Q 6.3 ############################
# Load the model
myNet = resnet50(weights="IMAGENET1K_V1")

# Parameters
data_dir = "../data/Imagenet32_val/"
max_iters = 100
batch_size = 64
numworkers = 2
lossf = nn.CrossEntropyLoss()
my_class = 123 # the chosen class to evaluate on

# Dataloaders
# ds = load_dataset("imagenet-1k")
# val_ds = ds["val"]
# print(ds)
# breakpoint()

# Dataloaders


# Dataloaders
# with open(data_dir + "val_data", 'rb') as f:
#     val_data = pickle.load(f)

# x = val_data["data"]
# y = val_data["labels"]
# # print(max(y,key=y.count))
# x = np.dstack((x[:, :1024], x[:, 1024:2048], x[:, 2048:]))
# x = x.reshape((x.shape[0], 32, 32))

# idx = [i for i in range(len(y)) if y[i] == my_class]
# x = [x[i] for i in idx]
# y = [y[i] for i in idx]
# print(x[0])
# print(y)
# breakpoint()

# y = [target-1 for target in y]

# tensor_x = torch.Tensor(x) # transform to torch tensor
# tensor_y = torch.Tensor(y).type(torch.LongTensor)

# my_dataset = TensorDataset(tensor_x, tensor_y) # create your datset
# test_loader = DataLoader(my_dataset, batch_size=batch_size) # create your dataloader

# Dataloaders
# testset = torchvision.datasets.ImageNet(root='../data/Imagenet32_val/', split='val', train=False, download=True, transform=transform)

# test_loader = torch.utils.data.DataLoader(testset,
#                                          shuffle=False, num_workers=2)

# Test with the validation dataset
test_acc, _ = evaluate_model(myNet, test_loader, lossf, device, False)
print("Test accuracy: ", test_acc)