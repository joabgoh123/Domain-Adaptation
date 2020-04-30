import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets,transforms, models
import torch.nn as nn
import torch.nn.functional as F
import time
import functools

def mmd_linear(f_of_X, f_of_Y):
    delta = f_of_X - f_of_Y
    loss = torch.mean(delta @ delta.t())
    return loss

class Resnet(nn.Module):
    def __init__(self):
        super(Resnet, self).__init__()
        resnet50 = models.resnet50(pretrained=True)
        self.resnet50_features = nn.Sequential(*list(resnet50.children())[:-1])
        self.class_classifier = nn.Sequential()
        self.class_classifier.add_module('c_fc0', nn.Linear(2048, 256))


    def forward(self, x):
        features = self.resnet50_features(x)
        # flatten for dense layer
        features = features.view(-1, 2048)
        output = self.class_classifier(features)
        
        return output

class DDC(nn.Module):
    def __init__(self, num_classes=11):
        super(DDC, self).__init__()
        self.convNet = Resnet()
        self.fc = nn.Linear(256, num_classes)

        # initialize according to CORAL paper experiment
        self.fc.weight.data.normal_(0, 0.005)

    def forward(self, source, target):
        source = self.convNet(source)
        source = self.fc(source)

        target = self.convNet(target)
        target = self.fc(target)

        mmd_loss = mmd_linear(source,target)
        return source, target, mmd_loss
