import torch
import torch.nn as nn
from torch.autograd import Function, Variable
from functions import ReverseLayerF
import torchvision.models as models
class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            # nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x

alexnet = models.alexnet(pretrained=True)

class DANN(nn.Module):
    def __init__(self):
        super(DANN, self).__init__()
        self.feature = AlexNet().features

        self.class_classifier = nn.Sequential()
        self.class_classifier.add_module('c_fc0', nn.Linear(256 * 6 * 6, 4096))
        self.class_classifier.add_module('c_drop1', nn.Dropout2d())
        self.class_classifier.add_module('c_relu1', nn.ReLU(True))
        self.class_classifier.add_module('c_fc1', nn.Linear(4096, 65))

        self.domain_classifier = nn.Sequential()
        self.domain_classifier.add_module('c_fc0', nn.Linear(256 * 6 * 6, 4096))
        self.domain_classifier.add_module('d_relu1', nn.ReLU(True))
        self.domain_classifier.add_module('d_fc2', nn.Linear(4096, 2))
    
    def forward(self, input_data, alpha):
        feature = self.feature(input_data)
        print(feature.size())
        feature = feature.view(-1, 256*6*6)
        reverse_feature = ReverseLayerF.apply(feature, alpha)
        class_output = self.class_classifier(feature)
        domain_output = self.domain_classifier(reverse_feature)

        return class_output, domain_output

class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        resnet50 = models.resnet50(pretrained=True)
        self.resnet50_features = nn.Sequential(*list(resnet50.children())[:-1])
        self.class_classifier = nn.Sequential()
        self.class_classifier.add_module('c_fc0', nn.Linear(2048, 11))


        self.domain_classifier = nn.Sequential()
        self.domain_classifier.add_module('d_fc0', nn.Linear(2048, 2))

    def forward(self, input_data, alpha):
        feature = self.resnet50_features(input_data)
        feature = feature.view(-1, 2048)
        reverse_feature = ReverseLayerF.apply(feature, alpha)
        class_output = self.class_classifier(feature)
        domain_output = self.domain_classifier(reverse_feature)

        return class_output, domain_output