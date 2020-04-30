from torchvision import datasets,transforms
from PIL import Image
import torch


def image_loader(path):
    transformations = transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor()])
    image = Image.open(path)
    image = transformations(image).float()
    # image = Variable(image, requires_grad=True)
    image = image.unsqueeze(0)  #this is for VGG, may not be needed for ResNet
    return image  #assumes that you're using GPU
