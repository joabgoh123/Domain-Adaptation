import torch
import torch.nn as nn
from torch.autograd import Function, Variable
from functions import ReverseLayerF
from collections import OrderedDict
class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.features = nn.Sequential(OrderedDict([
            ('c1', nn.Conv2d(3, 6, kernel_size=(5, 5))),
            ('relu1', nn.ReLU()),
            ('s2', nn.MaxPool2d(kernel_size=(2, 2), stride=2)),
            ('c3', nn.Conv2d(6, 16, kernel_size=(5, 5))),
            ('relu3', nn.ReLU()),
            ('s4', nn.MaxPool2d(kernel_size=(2, 2), stride=2)),
            ('c5', nn.Conv2d(16, 120, kernel_size=(5, 5))),
            ('relu5', nn.ReLU())
        ]))

    #     self.fc = nn.Sequential(OrderedDict([
    #         ('f6', nn.Linear(120, 84)),
    #         ('relu6', nn.ReLU()),
    #     ]))

    #     self.fc_cls = nn.Sequential(OrderedDict([
    #         ('f7', nn.Linear(84, 10)),
    #         ('sig7', nn.LogSoftmax(dim=-1))
    #     ]))

    # def forward(self, x):
    #     x = self.features(x)
    #     x = x.view(x.size(0), 256 * 6 * 6)
    #     x = self.classifier(x)
    #     return x

class DANN(nn.Module):
    def __init__(self):
        super(DANN, self).__init__()
        self.feature = LeNet5().features

        self.class_classifier = nn.Sequential()
        self.class_classifier.add_module('c_fc0', nn.Linear(120, 84))
        self.class_classifier.add_module('c_relu1', nn.ReLU(True))
        self.class_classifier.add_module('c_drop1', nn.Dropout2d())
        self.class_classifier.add_module('c_fc2', nn.Linear(84, 10))
#         self.class_classifier.add_module('c_sig1', nn.LogSoftmax(dim=-1))

        self.domain_classifier = nn.Sequential()
        self.domain_classifier.add_module('c_fc0', nn.Linear(120, 84))
        self.domain_classifier.add_module('c_relu1', nn.ReLU(True))
        self.domain_classifier.add_module('c_drop1', nn.Dropout2d())
        self.domain_classifier.add_module('c_fc2', nn.Linear(84, 2))
#         self.domain_classifier.add_module('c_sig2', nn.LogSoftmax(dim=-1))
    
    def forward(self, input_data, alpha):
        feature = self.feature(input_data)
        feature = feature.view(-1, 120)
        reverse_feature = ReverseLayerF.apply(feature, alpha)
        class_output = self.class_classifier(feature)
        domain_output = self.domain_classifier(reverse_feature)

        return class_output, domain_output
