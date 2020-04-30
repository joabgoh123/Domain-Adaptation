import sys, os
sys.path.append("C:/Users/Joab-PC/Desktop/FYP/GUI/DANN")
import Models
import torch
from utilities import image_loader

class DannModel():
    def __init__(self, device):
        self.device = device
        self.model = Models.ResNet()
        self.model.to(self.device)
        self.model.load_state_dict(torch.load("C:/Users/Joab-PC/Desktop/FYP/GUI/DANN/DANN_ResNet_Darkdata_100e.pth", map_location="cuda:0"))

    def predict(self, image_path):
        self.model.eval()
        classes = ["Bicycle", "Boat", "Bottle", "Bus", "Car", "Cat", "Chair",
            "Cup", "Dog", "Motorbike", "Table"]
        image = image_loader(image_path)
        image = image.to(self.device)
        output, _ = self.model(image, 0)
        pred = torch.argmax(output,1).item()

        softmax = torch.nn.Softmax(dim=1)
        prob = softmax(output).squeeze()


        return classes[pred], prob[pred].item()