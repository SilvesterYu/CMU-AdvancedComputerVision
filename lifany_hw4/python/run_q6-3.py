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
from torchvision.datasets import ImageFolder
import cv2
from PIL import Image 

if torch.cuda.is_available():
    device = 'cuda:0'  
else:
    device = 'cpu'
print('GPU State:', device)

################################ Q 6.3 ############################
# Load the model
myNet = resnet50(pretrained=True)

# Parameters
data_dir = "custom_dataq6/goose"
batch_size = 16
numworkers = 2
lossf = nn.CrossEntropyLoss()
my_class = 99 # the chosen class to evaluate on

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

val_transform = T.Compose([
    T.Resize(224),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
  ])

# Dataloader
val_dset = ImageFolder(data_dir, transform=val_transform)
val_loader = DataLoader(val_dset,
                    batch_size=batch_size,
                    shuffle=False, 
                num_workers=numworkers)

# Test with the validation dataset
test_acc, _ = evaluate_model(myNet, val_loader, lossf, device, False, 99)
print("Test accuracy on validation dataset: ", test_acc)

# ########################## custom video ###################3######
# Parameters
# custom_data_dir = "custom_dataq6"

# # Process video
# video_dir = custom_data_dir + '/video/'
# video_output_dir = custom_data_dir + '/images/'

# # Open a video file
# video_capture = cv2.VideoCapture(video_dir+f"goose.mp4")

# frame_count = 0
# while video_capture.isOpened():
#     ret, frame = video_capture.read()
#     if not ret:
#         break
#     cv2.imwrite(video_output_dir+f"frame_{frame_count}.jpg", frame)
#     frame_count += 1

# video_capture.release()
# cv2.destroyAllWindows()

# # Dataloader
# custom_dset = ImageFolder(video_output_dir, transform=val_transform)
# custom_loader = DataLoader(custom_dset,
#                     batch_size=batch_size,
#                     shuffle=False, 
#                 num_workers=numworkers)

# # Test with the custom video dataset
# test_acc, _ = evaluate_model(myNet, custom_loader, lossf, device, False, 99)
# print("Test accuracy on custom video dataset: ", test_acc)

