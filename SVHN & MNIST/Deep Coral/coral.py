import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets,transforms
import torch.nn as nn
import torch.nn.functional as F
import time
import functools
from collections import OrderedDict
import utils
import coral
def coral_loss(source,target):
    d = source.data.shape[1]

    # source covariance
    xm = torch.mean(source, 0, keepdim=True) - source
    xc = xm.t() @ xm

    # target covariance
    xmt = torch.mean(target, 0, keepdim=True) - target
    xct = xmt.t() @ xmt

    #frobenius norm between source and target
    loss = torch.norm(xct-xc)
    loss = loss ** 2
    loss = loss/(4*d*d)

    # loss = torch.norm(xc - xct)
    # loss = loss / (4*d*d)

    return loss





class LeNet5(nn.Module):
    """
    Input - 1x32x32
    C1 - 6@28x28 (5x5 kernel)
    tanh
    S2 - 6@14x14 (2x2 kernel, stride 2) Subsampling
    C3 - 16@10x10 (5x5 kernel, complicated shit)
    tanh
    S4 - 16@5x5 (2x2 kernel, stride 2) Subsampling
    C5 - 120@1x1 (5x5 kernel)
    F6 - 84
    tanh
    F7 - 10 (Output)
    """
    def __init__(self):
        super(LeNet5, self).__init__()
        self.convnet = nn.Sequential(OrderedDict([
            ('c1', nn.Conv2d(3, 6, kernel_size=(5, 5))),
            ('relu1', nn.ReLU()),
            ('s2', nn.MaxPool2d(kernel_size=(2, 2), stride=2)),
            ('c3', nn.Conv2d(6, 16, kernel_size=(5, 5))),
            ('relu3', nn.ReLU()),
            ('s4', nn.MaxPool2d(kernel_size=(2, 2), stride=2)),
            ('c5', nn.Conv2d(16, 120, kernel_size=(5, 5))),
            ('relu5', nn.ReLU())
        ]))

        self.fc = nn.Sequential(OrderedDict([
            ('f6', nn.Linear(120, 84)),
            ('relu6', nn.ReLU()),
        ]))


    
    def forward(self, x):
        x = self.convnet(x)
        x = x.view(-1,120)
        x = self.fc(x)
        return x

class DeepCORAL(nn.Module):
    def __init__(self, num_classes=10):
        super(DeepCORAL, self).__init__()
        self.convNet = LeNet5()
        self.fc = nn.Linear(84, num_classes)

        # initialize according to CORAL paper experiment
        self.fc.weight.data.normal_(0, 0.005)

    def forward(self, source, target):
        source = self.convNet(source)
        source = self.fc(source)

        target = self.convNet(target)
        target = self.fc(target)
        return source, target
