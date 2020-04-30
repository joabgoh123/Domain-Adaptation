import sys, os
sys.path.append("C:/Users/Joab-PC/Desktop/FYP/GUI/DDC")
import torch
import DDC_resnet as DDC
from utilities import image_loader

class DDCModel():
    def __init__(self, device):
        self.device = device
        self.model = DDC.DDC()
        self.model.to(device)
        self.model.load_state_dict(torch.load("C:/Users/Joab-PC/Desktop/FYP/GUI/DDC/ddc_resnet_100e_lambda_adj.pth", map_location="cuda:0"))
    
    def predict(self, image_path):
        self.model.eval()
        classes = ["Bicycle", "Boat", "Bottle", "Bus", "Car", "Cat", "Chair",
         "Cup", "Dog", "Motorbike", "Table"]
        image = image_loader(image_path)
        image = image.to(self.device)
        output, _ , _= self.model(image, image)
        pred = torch.argmax(output,1).item()

        softmax = torch.nn.Softmax(dim=1)
        prob = softmax(output).squeeze()

        return classes[pred], prob[pred].item()
        