################################################################################
# CSE 190: Programming Assignment 3
# Fall 2018
# Code author: Songlin Li
#
# Filename: CNN_Detector.py
# 
# Description: 
# 
# This file contains the starter code for the baseline architecture you will use
# to get a little practice with PyTorch and compare the results of with your 
# improved architecture. 
#
# Be sure to fill in the code in the areas marked #TODO.
################################################################################

import numpy as np
import os

# PyTorch and neural network imports
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as func
import torch.nn.init as torch_init
import torch.optim as optim

# Data utils and dataloader
import torchvision
from torchvision import transforms, utils

class CNN_Detector(nn.Module):
    """ A basic convolutional neural network model for baseline comparison. 
    
    Consists of three Conv2d layers, followed by one 4x4 max-pooling layer, 
    and 2 fully-connected (FC) layers:
    
    conv1 -> conv2 -> maxpool1 -> conv3 -> conv4 -> maxpool2 -> conv5  -> fc1 -> fc2 (outputs)
    
    Make note: 
    - Inputs are expected to be grayscale images (how many channels does this imply?)
    - The Conv2d layer uses a stride of 1 and 0 padding by default
    """
    
    def __init__(self):
        super(CNN_Detector, self).__init__()
        
        # conv1: 1 input channel, 6 output channels, [8x8] kernel size
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=8)
        self.conv1_normed = nn.BatchNorm2d(6)
        torch_init.xavier_normal_(self.conv1.weight)        
        
        # conv2: 6 input channel, 12 output channels, [7x7] kernel size 
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=7)
        self.conv2_normed = nn.BatchNorm2d(12)
        torch_init.xavier_normal_(self.conv2.weight)

        # first pooling layer [3x3] 
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=3)
        
        # conv3 12 input channels, 10 output channels, [8x8] kernel
        self.conv3 = nn.Conv2d(in_channels=12, out_channels=10, kernel_size=8)
        self.conv3_normed = nn.BatchNorm2d(10)
        torch_init.xavier_normal_(self.conv3.weight)

        self.conv4 = nn.Conv2d(in_channels=10, out_channels=8, kernel_size=7)
        self.conv4_normed = nn.BatchNorm2d(8)
        torch_init.xavier_normal_(self.conv4.weight)

        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=3)

        self.conv5 = nn.Conv2d(in_channels=8, out_channels=6, kernel_size=6)
        self.conv5_normed = nn.BatchNorm2d(6)
        torch_init.xavier_normal_(self.conv5.weight)

        
        # Define 2 fully connected layers:
        self.fc1 = nn.Linear(in_features=103 * 103 * 6, out_features=128)
        self.fc1_normed = nn.BatchNorm1d(128)
        torch_init.xavier_normal_(self.fc1.weight)

        self.fc2 = nn.Linear(in_features=128, out_features=14).cuda()
        torch_init.xavier_normal_(self.fc2.weight)



    def forward(self, batch):
        """Pass the batch of images through each layer of the network, applying 
        non-linearities after each layer.
        
        Note that this function *needs* to be called "forward" for PyTorch to 
        automagically perform the forward pass. 
        
        Params:
        -------
        - batch: (Tensor) An input batch of images

        Returns:
        --------
        - logits: (Variable) The output of the network
        """
        
        # Apply first convolution, followed by ReLU non-linearity; 
        # use batch-normalization on its outputs
        batch = func.relu(self.conv1_normed(self.conv1(batch)))
        batch = func.relu(self.conv2_normed(self.conv2(batch)))
        batch = self.pool1(batch)
        
        batch = func.relu(self.conv3_normed(self.conv3(batch)))
        batch = func.relu(self.conv4_normed(self.conv4(batch)))
        batch = self.pool2(batch)
        
        batch = func.relu(self.conv5_normed(self.conv5(batch)))
        
        # Reshape the output of the conv3 to pass to fully-connected layer
        batch = batch.view(-1, self.num_flat_features(batch))
        
        # Connect the reshaped features of the pooled conv3 to fc1
        batch = func.relu(self.fc1_normed(self.fc1(batch)))
        
        # Connect fc1 to fc2 - this layer is slightly different than the rest (why?)
        batch = self.fc2(batch)

        # Return the class predictions
        return batch
    

    def num_flat_features(self, inputs):
        
        # Get the dimensions of the layers excluding the inputs
        size = inputs.size()[1:]
        # Track the number of features
        num_features = 1
        
        for s in size:
            num_features *= s
        
        return num_features
