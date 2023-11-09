import torch
import torchvision
from torchvision.models import squeezenet1_1, SqueezeNet1_1_Weights
import numpy as np
import scipy
from nnq6 import *

################################ Q 6.2 ############################
# Load the pretrained network squeezenet1_1
myNet = squeezenet1_1(SqueezeNet1_1_Weights)
print(myNet)

