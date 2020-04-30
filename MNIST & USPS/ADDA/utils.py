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
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True, num_workers = 6)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=True, num_workers = 6)

        #Check data Sizes
        print("Number of Training Examples : {:d} ".format(len(train_data)))
        print("Number of Test Examples : {:d} ".format(len(test_data)))
        print("Shape of data: " + str(train_data[0][0].size()))

        #Data Loaders
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=256, shuffle=True, num_workers = 16)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=256, shuffle=True, num_workers = 16)

        return train_loader, test_loader

    if dataset.upper() == "MNIST":
        transform = transforms.Compose([transforms.ToTensor()])
        train_data = datasets.MNIST('~\\Datasets',download=True, train=True, transform=transform)
        test_data = datasets.MNIST('~\\Datasets',download=True, train=False, transform=transform)
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=256, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=256, shuffle=True)

        return train_loader,test_loader

    if dataset.upper() == "USPS":
        train_loader = get_usps(train=True)
        test_loader = get_usps(train=False)

        return train_loader, test_loader

def mmd_linear(f_of_X, f_of_Y):
    delta = f_of_X - f_of_Y
    loss = torch.mean(torch.mm(delta, torch.transpose(delta, 0, 1)))
    return loss



"""Dataset setting and data loader for USPS.

Modified from
https://github.com/mingyuliutw/CoGAN/blob/master/cogan_pytorch/src/dataset_usps.py
"""

import gzip
import os
import pickle
import urllib

import numpy as np
import torch
import torch.utils.data as data
from torchvision import datasets, transforms

import params


class USPS(data.Dataset):
    """USPS Dataset.

    Args:
        root (string): Root directory of dataset where dataset file exist.
        train (bool, optional): If True, resample from dataset randomly.
        download (bool, optional): If true, downloads the dataset
            from the internet and puts it in root directory.
            If dataset is already downloaded, it is not downloaded again.
        transform (callable, optional): A function/transform that takes in
            an PIL image and returns a transformed version.
            E.g, ``transforms.RandomCrop``
    """

    url = "https://raw.githubusercontent.com/mingyuliutw/CoGAN/master/cogan_pytorch/data/uspssample/usps_28x28.pkl"

    def __init__(self, root, train=True, transform=None, download=False):
        """Init USPS dataset."""
        # init params
        self.root = os.path.expanduser(root)
        self.filename = "usps_28x28.pkl"
        self.train = train
        # Num of Train = 7438, Num ot Test 1860
        self.transform = transform
        self.dataset_size = None

        # download dataset.
        if download:
            self.download()
        if not self._check_exists():
            raise RuntimeError("Dataset not found." +
                               " You can use download=True to download it")

        self.train_data, self.train_labels = self.load_samples()
        if self.train:
            total_num_samples = self.train_labels.shape[0]
            indices = np.arange(total_num_samples)
            np.random.shuffle(indices)
            self.train_data = self.train_data[indices[0:self.dataset_size], ::]
            self.train_labels = self.train_labels[indices[0:self.dataset_size]]
        self.train_data *= 255.0
        self.train_data = self.train_data.transpose(
            (0, 2, 3, 1))  # convert to HWC

    def __getitem__(self, index):
        """Get images and target for data loader.

        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, label = self.train_data[index, ::], self.train_labels[index]
        if self.transform is not None:
            img = self.transform(img)
        label = torch.LongTensor([np.int64(label).item()])
        # label = torch.FloatTensor([label.item()])
        return img, label

    def __len__(self):
        """Return size of dataset."""
        return self.dataset_size

    def _check_exists(self):
        """Check if dataset is download and in right place."""
        return os.path.exists(os.path.join(self.root, self.filename))

    def download(self):
        """Download dataset."""
        filename = os.path.join(self.root, self.filename)
        dirname = os.path.dirname(filename)
        if not os.path.isdir(dirname):
            os.makedirs(dirname)
        if os.path.isfile(filename):
            return
        print("Download %s to %s" % (self.url, os.path.abspath(filename)))
        urllib.request.urlretrieve(self.url, filename)
        print("[DONE]")
        return

    def load_samples(self):
        """Load sample images from dataset."""
        filename = os.path.join(self.root, self.filename)
        f = gzip.open(filename, "rb")
        data_set = pickle.load(f, encoding="bytes")
        f.close()
        if self.train:
            images = data_set[0][0]
            labels = data_set[0][1]
            self.dataset_size = labels.shape[0]
        else:
            images = data_set[1][0]
            labels = data_set[1][1]
            self.dataset_size = labels.shape[0]
        return images, labels


def get_usps(train):
    """Get USPS dataset loader."""
    # image pre-processing
    # pre_process = transforms.Compose([transforms.ToTensor(),
    #                                   transforms.Normalize(
    #                                       mean=params.dataset_mean,
    #                                       std=params.dataset_std)])

    pre_process = transforms.Compose([transforms.ToTensor()])


    # dataset and data loader
    usps_dataset = USPS(root=params.data_root,
                        train=train,
                        transform=pre_process,
                        download=True)

    usps_data_loader = torch.utils.data.DataLoader(
        dataset=usps_dataset,
        batch_size=params.batch_size,
        shuffle=True)

    return usps_data_loader


