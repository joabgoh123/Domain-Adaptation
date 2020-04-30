import torch
import numpy as np
from torchvision import datasets,transforms, models
import torch.nn as nn
import torch.nn.functional as F
from utilities import image_loader

class ResNet(nn.Module):
    def __init__(self, num_classes=11):
        super(ResNet, self).__init__()
        self.resnet50 = models.resnet50(pretrained=True)
        self.cls_layer = nn.Linear(1000,num_classes)
        
    def forward(self,x):
        x = self.resnet50(x)
        x = self.cls_layer(x)
        return x


class SourceModel():
    def __init__(self, device):
        self.device = device
        self.model = ResNet()
        self.model.to(self.device)
        self.model.load_state_dict(torch.load("C:/Users/Joab-PC/Desktop/FYP/GUI/ResNet50_NoTransferLearning.pth", map_location='cuda: 0'))

    def predict(self, image_path):
        self.model.eval()
        classes = ["Bicycle", "Boat", "Bottle", "Bus", "Car", "Cat", "Chair",
         "Cup", "Dog", "Motorbike", "Table"]
        image = image_loader(image_path)
        image = image.to(self.device)
        output = self.model(image)
        pred = torch.argmax(output,1).item()

        softmax = torch.nn.Softmax(dim=1)
        prob = softmax(output).squeeze()

        return classes[pred], prob[pred].item()