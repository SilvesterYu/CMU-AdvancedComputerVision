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

# Load original data
train_data = scipy.io.loadmat("../data/nist36_train.mat")
valid_data = scipy.io.loadmat("../data/nist36_valid.mat")
test_data = scipy.io.loadmat("../data/nist36_test.mat")

train_x, train_y = train_data["train_data"], train_data["train_labels"]
valid_x, valid_y = valid_data["valid_data"], valid_data["valid_labels"]
test_x, test_y = test_data["test_data"], test_data["test_labels"]

trainLoader = np2loader(train_x, train_y, batchsize=64)
validLoader = np2loader(valid_x, valid_y, shuffling=False)
testLoader = np2loader(test_x, test_y, shuffling=False)

trainLoaderCNN = np2loader(train_x.reshape((len(train_x), 1, 32, 32)), train_y, batchsize=64)

validLoaderCNN = np2loader(valid_x.reshape((len(valid_x), 1, 32, 32)), valid_y)
################################## Q6.1.1 #######################################
# # Call the network
# myNet = Net()

# # Parameters
max_iters = 200
learning_rate = 1e-1
lossf = nn.CrossEntropyLoss()
# optimizer = optim.SGD(myNet.parameters(), lr=learning_rate)
# fname = 'q6_fully_connected.pth'

# # Training loop
# training_loop(myNet, trainLoader, validLoader, device, max_iters, learning_rate, lossf, optimizer, fname)

# # Test
# myNet.load_state_dict(torch.load(fname))
# test_acc, _ = evaluate_model(myNet, testLoader, lossf, device)
# print("Test accuracy: ", test_acc)

################################## Q6.1.2 #######################################
fname = 'test.pth'
myCNN = CNN().to(device)
optimizer = optim.SGD(myCNN.parameters(), lr=learning_rate)

training_loop(myCNN, trainLoaderCNN, validLoaderCNN, device, max_iters, learning_rate, lossf, optimizer, fname)

# def train(model, device, train_loader, optimizer, epoch):
#     model.train()
#     for batch_idx, (data, target) in enumerate(train_loader):
#         data, target = data.to(device), target.to(device)
#         optimizer.zero_grad()
#         output = model(data)
#         loss = F.nll_loss(output.log(), target)
#         loss.backward()
#         optimizer.step()
#         if batch_idx % 100 == 0:
#             print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
#                 epoch, batch_idx * len(data), len(train_loader.dataset),
#                 100. * batch_idx / len(train_loader), loss.item()))

# def test(model, device, test_loader):
#     model.eval()
#     test_loss = 0
#     correct = 0
#     with torch.no_grad():
#         for data, target in test_loader:
#             data, target = data.to(device), target.to(device)
#             output = model(data)
#             test_loss += F.nll_loss(output.log(), target).item() # sum up batch loss
#             pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
#             correct += pred.eq(target.view_as(pred)).sum().item()

#     test_loss /= len(test_loader.dataset)
#     print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
#         test_loss, correct, len(test_loader.dataset),
#     100. * correct / len(test_loader.dataset)))

# optimizer = optim.SGD(myCNN.parameters(), lr=0.01, momentum=0.5)

# for epoch in range(1, max_iters + 1):
#     train(myCNN, device, trainLoaderCNN, optimizer, epoch)
#     test(myCNN, device, testLoader)