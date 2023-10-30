import numpy as np


# use for a "no activation" layer
def linear(x):
    return x


def linear_deriv(post_act):
    return np.ones_like(post_act)


def tanh(x):
    return np.tanh(x)


def tanh_deriv(post_act):
    return 1 - post_act**2


def relu(x):
    return np.maximum(x, 0)


def relu_deriv(x):
    return (x > 0).astype(float)


### Sample dataloader For Q.6.1 ####

# import torch
# import torch.nn as nn
# from torch.utils.data import Dataset, DataLoader


# class NIST36_Data(torch.utils.data.Dataset):
#     def __init__(self, type):
#         self.type = type
#         self.data = scipy.io.loadmat(f"../data/nist36_{type}.mat")
#         self.inputs, self.one_hot_target = (
#             self.data[f"{self.type}_data"],
#             self.data[f"{self.type}_labels"],
#         )
#         self.target = np.argmax(self.one_hot_target, axis=1)

#     def __len__(self):
#         return self.inputs.shape[0]

#     def __getitem__(self, index):
#         inputs = torch.from_numpy(self.inputs[index]).type(torch.float32)
#         target = torch.tensor(self.target[index]).type(torch.LongTensor)
#         return inputs, target


###### Example usage #############

# train_data = NIST36_Data(type="train")
# valid_data = NIST36_Data(type="valid")
# test_data = NIST36_Data(type="test")

# trainloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
# testloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

###################################
