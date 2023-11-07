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

# # Dataloaders
# trainSet = datasets.MNIST(root='MNIST', download=True, train=True, transform=transform)
# testSet = datasets.MNIST(root='MNIST', download=True, train=False, transform=transform)
# trainLoader = dset.DataLoader(trainSet, batch_size=64, shuffle=True)
# testLoader = dset.DataLoader(testSet, batch_size=64, shuffle=False)

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
learning_rate = 1e-1
lossf = nn.CrossEntropyLoss()
optimizer = optim.SGD(myNet.parameters(), lr=learning_rate)
fname = 'q6_fully_connected'

# Training loop
training_loop(myNet, trainLoader, device, max_iters, learning_rate, lossf, optimizer, fname)





















# for itr in range(max_iters):
#     total_loss = 0.0
#     total_correct = 0.0
#     total_instances = 0

#     for times, data in enumerate(trainLoader):
#         # print(data)
#         inputs, labels = data[0].to(device), data[1].to(device)
#         inputs = inputs.view(inputs.shape[0], -1)

#         # Zero the parameter gradients
#         optimizer.zero_grad()

#         # Foward, backward, optimize
#         outputs = myNet(inputs)
#         loss = lossf(outputs, labels)
#         loss.backward()
#         optimizer.step()

#         # Total loss
#         total_loss += loss.item()

#         with torch.no_grad():
#             # average accuracy
#             classifications = torch.argmax(myNet(inputs), dim=1)
#             # labels = torch.argmax(labels, dim=1)
#             correct_predictions = sum(classifications==labels).item()
#             total_correct+=correct_predictions
#             total_instances+=len(inputs)
        
#     accuracy = round(total_correct/total_instances, 4)

#     #     # since we're not training, we don't need to calculate the gradients for our outputs
#     #     with torch.no_grad():
#     #         outputs = myNet(inputs)
#     #         # the class with the highest energy is what we choose as prediction
#     #         _, predicted = torch.max(outputs.data, 1)
#     #         total_instances += labels.size(0)
#     #         total_correct += (predicted == labels).sum().item()
#     # accuracy = total_correct / total_instances
    
#     if itr % 10 == 0:
        
#         print(
#             "itr: {:02d} \t loss: {:.2f} \t acc : {:.2f}".format(
#                 itr, total_loss, accuracy
#             )
#         )

# # save the weights
# torch.save(myNet.state_dict(), 'q6_fully_connected.pth')



