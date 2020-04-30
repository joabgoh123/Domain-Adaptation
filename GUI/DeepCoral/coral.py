import sys, os
sys.path.append("C:/Users/Joab-PC/Desktop/FYP/GUI/DeepCoral")
import coral_resnet
import torch
from utilities import image_loader

class CoralModel():
    def __init__(self, device):
        self.device = device
        self.model = coral_resnet.DeepCORAL()
        self.model.to(self.device)
        self.model.load_state_dict(torch.load("C:/Users/Joab-PC/Desktop/FYP/GUI/DeepCoral/coral_resnet_100e.pth", map_location="cuda:0"))
    
    def predict(self, image_path):
        self.model.eval()
        classes = ["Bicycle", "Boat", "Bottle", "Bus", "Car", "Cat", "Chair",
         "Cup", "Dog", "Motorbike", "Table"]
        image = image_loader(image_path)
        image = image.to(self.device)
        output, _ = self.model(image, image)
        pred = torch.argmax(output,1).item()

        softmax = torch.nn.Softmax(dim=1)
        prob = softmax(output).squeeze()

        return classes[pred], prob[pred].item()
