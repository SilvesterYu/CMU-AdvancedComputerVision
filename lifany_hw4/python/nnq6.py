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

# TUTORIAL: https://clay-atlas.com/us/blog/2021/04/22/pytorch-en-tutorial-4-train-a-model-to-classify-mnist/

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
    
# TUTORIAL: https://medium.com/@nutanbhogendrasharma/pytorch-convolutional-neural-network-with-mnist-dataset-4e8a4265e118

# for Q6.2.1
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        # First 2D convolutional layer, taking in 1 input channel (image),
        # outputting 32 convolutional features, with a square kernel size of 3
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        # Second 2D convolutional layer, taking in the 32 input layers,
        # outputting 64 convolutional features, with a square kernel size of 3
        self.conv2 = nn.Conv2d(32, 64, 3, 1)

        # Designed to ensure that adjacent pixels are either all 0s or all active
        # with an input probability
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)

        # First fully connected layer
        self.fc1 = nn.Linear(9216, 128)
        # Second fully connected layer that outputs our 10 labels
        self.fc2 = nn.Linear(128, 10)

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

def training_loop(myNet, trainLoader, validLoader, device, max_iters, learning_rate, lossf, optimizer, fname):
    # Training loop
    # Initialize the network
    myNet = myNet.to(device)
    print(myNet)
    train_loss_list, train_acc_list, val_loss_list, val_acc_list = [], [], [], []
    for itr in range(max_iters):
        accuracy, train_loss = evaluate_model(myNet, trainLoader, lossf, device)
        train_acc_list.append(accuracy)
        train_loss_list.append(train_loss)

        val_accuracy, val_loss = evaluate_model(myNet, validLoader, lossf, device)
        val_acc_list.append(val_accuracy)
        val_loss_list.append(val_loss)

        myNet.train()

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

            with torch.no_grad():
                # average accuracy
                classifications = torch.argmax(myNet(inputs), dim=1)
                # labels = torch.argmax(labels, dim=1)
                correct_predictions = sum(classifications==labels).item()
                total_correct+=correct_predictions
                total_instances+=len(inputs)
            
        ave_accuracy = round(total_correct/total_instances, 4)
        
        if itr % 10 == 0:
            
            print(
                "itr: {:02d} \t loss: {:.2f} \t acc : {:.2f} \t eval_acc : {:.2f}".format(
                    itr, total_loss, ave_accuracy,val_accuracy
                )
            )

    # save the weights
    torch.save(myNet.state_dict(), fname)

    # visualize
    plot_train_valid(train_acc_list, val_acc_list, "accuracy")
    plot_train_valid(train_loss_list, val_loss_list, "average loss")

def evaluate_model(myNet, validLoader, lossf, device):
    myNet.eval()
    total_loss = 0.0
    total_correct = 0.0
    total_instances = 0
    for times, data in enumerate(validLoader):
        inputs, labels = data[0].to(device), data[1].to(device)
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


