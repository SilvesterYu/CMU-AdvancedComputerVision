import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

####################################################################################################
######################################## NEURAL NETWORKS ###########################################
####################################################################################################

# for Q6.1.1
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.main = nn.Sequential(
            nn.Linear(1024, 64),
            nn.Sigmoid(),
            nn.Linear(64, 36),
            #nn.Softmax()
        )

    def forward(self, X):
        return self.main(X)
    
# for Q6.1.2
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 10, kernel_size=5),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(10, 20, kernel_size=5),
            nn.Dropout(),
            nn.MaxPool2d(2),
            nn.ReLU(),
        )

        self.fc_layers = nn.Sequential(
            nn.Linear(500, 64),
            nn.Sigmoid(),
            nn.Linear(64, 36),
            #nn.Softmax()
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = torch.flatten(x, 1)
        x = self.fc_layers(x)
        return x
    
# for Q6.1.3
class CNNcifar(nn.Module):
    def __init__(self):
        super(CNNcifar, self).__init__()

        self.conv_layers = nn.Sequential(
            # nn.Conv2d(3, 32, kernel_size=5),
            # nn.MaxPool2d(2, 2),
            # nn.ReLU(),
            # nn.Conv2d(32, 64, kernel_size=5),
            # nn.Dropout(0.2),
            # nn.MaxPool2d(2, 2),
            # nn.ReLU(),
            # nn.BatchNorm2d(64) 

            # sees 32x32x3 image tensor
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            # sees 16x16x16 tensor
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            # sees 8x8x32 tensor
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)

            # nn.Conv2d(3, 32, 5, padding=1),
            # nn.MaxPool2d(2, 2),
            # nn.Conv2d(32, 64, 5, padding=1),
            # nn.MaxPool2d(2, 2)
        )

        self.fc_layers = nn.Sequential(
            nn.Linear(2304, 500),
            nn.ReLU(),
            nn.Linear(500, 10),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = torch.flatten(x, 1)
        x = self.fc_layers(x)
        return x

####################################################################################################
######################################## HELPER FUNCTIONS ##########################################
####################################################################################################

# convert numpy array to dataloader
def np2loader(X, y, batchsize=128, shuffling=True):
    y = y.argmax(axis=1)
    X = torch.from_numpy(np.float32(X))
    y = torch.from_numpy(y)
    data = torch.utils.data.TensorDataset(X, y)
    if batchsize != None:
        loader = torch.utils.data.DataLoader(data, batch_size=batchsize, shuffle=shuffling)
    else:
        loader = torch.utils.data.DataLoader(data, shuffle=shuffling)
    return loader

def training_loop(myNet, trainLoader, validLoader, device, max_iters, learning_rate, lossf, optimizer, fname, flatten=True):
    # Training loop
    # Initialize the network
    myNet = myNet.to(device)
    print(myNet)
    train_loss_list, train_acc_list, val_loss_list, val_acc_list = [], [], [], []
    for itr in range(max_iters):
        myNet.train()

        total_loss = 0.0
        total_correct = 0.0
        total_instances = 0

        for times, data in enumerate(trainLoader):
            #print(data)
            inputs, labels = data[0].to(device), data[1].to(device)
            if flatten:
                inputs = inputs.view(inputs.shape[0], -1)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Foward, backward, optimize
            outputs = myNet(inputs)
            # print(outputs.shape, labels.shape)
            # print(outputs)
            # print(labels)
            loss = lossf(outputs, labels)
            #breakpoint()
            loss.backward()
            optimizer.step()

            total_loss += loss

        train_accuracy, train_loss = evaluate_model(myNet, trainLoader, lossf, device, flatten)
        train_acc_list.append(train_accuracy)
        train_loss_list.append(train_loss)

        val_accuracy, val_loss = evaluate_model(myNet, validLoader, lossf, device, flatten)
        val_acc_list.append(val_accuracy)
        val_loss_list.append(val_loss)
        
        if itr % 10 == 0:
            print(
                "itr: {:02d} \t loss: {:.2f} \t acc : {:.2f} \t eval_acc : {:.2f}".format(
                    itr, total_loss, train_accuracy, val_accuracy
                )
            )

    # save the weights
    torch.save(myNet.state_dict(), fname)

    # visualize
    plot_train_valid(train_acc_list, val_acc_list, "accuracy")
    plot_train_valid(train_loss_list, val_loss_list, "average loss")
    plot_train(train_acc_list, "accuracy")
    plot_train(train_loss_list, "average loss")

def evaluate_model(myNet, dataLoader, lossf, device, flatten=True):
    myNet.eval()
    total_loss = 0.0
    total_correct = 0.0
    total_instances = 0
    for times, data in enumerate(dataLoader):
        inputs, labels = data[0].to(device), data[1].to(device)
        if flatten:
            inputs = inputs.view(inputs.shape[0], -1)
        outputs = myNet(inputs)
        loss = lossf(outputs, labels)
        # Total loss
        total_loss += loss.item()
        with torch.no_grad():
            # average accuracy
            classifications = torch.argmax(myNet(inputs), dim=1)
            # labels = torch.argmax(labels, dim=1)
            correct_predictions = sum(classifications==labels).item()
            total_correct+=correct_predictions
            total_instances+=len(inputs)
    accuracy = round(total_correct/total_instances, 4)
    return accuracy, total_loss/total_instances

# Plot train and valid loss / accuracies
def plot_train_valid(train_data, valid_data, datatype):
    plt.plot(range(len(train_data)), train_data, label="training")
    plt.plot(range(len(valid_data)), valid_data, label="validation")
    plt.xlabel("epoch")
    plt.ylabel(datatype)
    plt.xlim(0, len(train_data) - 1)
    plt.ylim(0, None)
    plt.legend()
    plt.grid()
    plt.show()

# Plot train and loss / accuracies
def plot_train(train_data, datatype):
    plt.plot(range(len(train_data)), train_data, label="training")
    plt.xlabel("epoch")
    plt.ylabel(datatype)
    plt.xlim(0, len(train_data) - 1)
    plt.ylim(0, None)
    plt.legend()
    plt.grid()
    plt.show()