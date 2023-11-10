# import numpy as np

# # mask for row-wise maximum
# a = np.array([[0, 1],
#              [2, 3],
#             [4, 5],
#             [0, 0],
#         [6, 7],
#           [9, 8],
#           [0, 0]])

# b = (a == a.max(axis=1)[:,None]).astype(int)
# print(b)

# # number of zero rows
# zerorows = np.sum(~a.any(1))
# print(zerorows)

# print("-"*10)
# arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
# np.split(arr,3)

# X = np.array([[1, 2],
# [3, 4],
# [5, 6],
# [7, 8]])
# W = np.array([[0.1, 0.2, 0.3],
# [1.1, 1.2, 1.3]])
# b = np.array([5, 6, 7])
# pre_act = np.matmul(X, W) + b
# print("XW", np.matmul(X, W))
# print("pre", pre_act)
# print("split x", np.split(X, 2))

# print("+++++")
# test = np.array([[1, 2, 3, 4],
# [5, 6, 7, 8],
# [9, 10, 11, 12]])
# thisshape = test.shape
# test = test.flatten()

# print(test)
# print(len(test.shape))
# test = test.reshape(thisshape)
# print(test)
# print(len(test.shape))

# points = [0.1, 0.31,  0.32, 0.45, 0.35, 0.40, 0.5 ]

# clusters = []
# eps = 0.2
# points_sorted = sorted(points)
# sort_index = numpy.argsort(points)
# curr_point = points_sorted[0]
# curr_cluster = [curr_point]
# for point in points_sorted[1:]:
#     if point <= curr_point + eps:
#         curr_cluster.append(point)
#     else:
#         clusters.append(curr_cluster)
#         curr_cluster = [point]
#     curr_point = point
# clusters.append(curr_cluster)
# print(clusters)

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import torch
import torchvision
from torchvision.models import squeezenet1_1, SqueezeNet1_1_Weights

from torchvision.datasets.utils import download_url
from torch.utils.data import random_split
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torchvision.transforms import ToTensor
from torch.utils.data.dataloader import DataLoader
import torch.nn as nn
import torch.nn.functional as F

# Load the pretrained network squeezenet1_1
myNet = squeezenet1_1(SqueezeNet1_1_Weights)
print(myNet)

# Parameters
data_dir = "../data/oxford-flowers17/"
max_iters = 50
learning_rate = 1e-1
batch_size = 64
numworkers = 2
lossf = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(myNet.parameters())
fname = "q6_flowers.pth"

transformer = torchvision.transforms.Compose(
    [  # Applying Augmentation
        torchvision.transforms.Resize((224, 224)),
        torchvision.transforms.RandomHorizontalFlip(p=0.5),
        torchvision.transforms.RandomVerticalFlip(p=0.5),
        torchvision.transforms.RandomRotation(30),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),
    ]
)

traindata = ImageFolder(data_dir + "train", transform=transformer)

valdata = ImageFolder(data_dir + "val", transform=transformer)

testdata = ImageFolder(data_dir + "test", transform=transformer)

