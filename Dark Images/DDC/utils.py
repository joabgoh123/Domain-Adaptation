import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets,transforms
import torch.nn as nn
import torch.nn.functional as F
import time
import functools

def mean_std(train_data):
    train_dataset = torch.utils.data.DataLoader(train_data, batch_size=len(train_data), shuffle=True)
    images, samples = iter(train_dataset).__next__()
    print("Train data mean: " + str(images.mean((0,2,3))))
    print("Train data std: " + str(images.std((0,2,3))))
    mean = images.mean((0,2,3))
    std = images.std((0,2,3))
    return mean,std

def load_data(dataset):
    if dataset.upper() == "SVHN":
        # mean = (0,0,0)
        # std = (1,1,1)
        # #Transforms
        transform = transforms.Compose([transforms.ToTensor()])

        # #Download Datasets
        # train_data = datasets.SVHN('~\\Datasets', download=True,transform=transform)
        # test_data = datasets.SVHN('~\\Datasets',download=True, split='test', transform=transform)


        # #Data Loaders
        # train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
        # test_loader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=True)

        # #Data Loaders
        # train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
        # test_loader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=True)

        # mean,std = mean_std(train_data)
        # #Transforms
        # transform = transforms.Compose([transforms.ToTensor(),
        #                             transforms.Normalize(mean,std)])

        #Download Datasets
        train_data = datasets.SVHN('~\\Datasets', download=True,transform=transform)
        test_data = datasets.SVHN('~\\Datasets',download=True, split='test', transform=transform)

        #Split train data to train and val

        #Data Loaders
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=True)

        #Check data Sizes
        print("Number of Training Examples : {:d} ".format(len(train_data)))
        print("Number of Test Examples : {:d} ".format(len(test_data)))
        print("Shape of data: " + str(train_data[0][0].size()))

        #Data Loaders
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=True)

        return train_loader, test_loader

    if dataset.upper() == "MNIST":
        transform = transforms.Compose([transforms.ToTensor()])
        train_data = datasets.MNIST('~\\Datasets',download=True, train=True, transform=transform)
        test_data = train_data = datasets.MNIST('~\\Datasets',download=True, train=True, transform=transform)
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=True)

        return train_loader,test_loader

def mmd_linear(f_of_X, f_of_Y):
    delta = f_of_X - f_of_Y
    loss = torch.mean(torch.mm(delta, torch.transpose(delta, 0, 1)))
    return loss



