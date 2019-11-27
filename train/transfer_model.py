# Part III. Transfer Learning using pre-trained models
#import necessary libraries for running the model

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import pandas as pd
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import time

# A. Load the pretained weight for resnet50
transfer_model = models.resnet50(pretrained=True)
#transfer_model

# B. Freeze the weights for the earlier convolutional layers (lower-level features)
'''Freeze all of the existing layers in the network
    by setting requires_grad to False'''

for name, param in transfer_model.named_parameters():
    if("bn" not in name):
        param.requires_grad = False

#C. Replace the fully connected layers with a classifier specific to this dataset
num_inputs = transfer_model.fc.in_features

# Add on classifier
transfer_model.fc = nn.Sequential(
    nn.Linear(num_inputs, 256), nn.ReLU(), nn.Dropout(0.4),
    nn.Linear(256, 6), nn.LogSoftmax(dim=1))

transfer_model.fc

total_params = sum(param.numel() for param in transfer_model.parameters())
print(f'There are {total_params:,} total parameters.')
total_trainable_params = sum(
    param.numel() for param in transfer_model.parameters() if param.requires_grad)
print(f'There are {total_trainable_params:,} training parameters.')

