#!/usr/bin/env python
# coding: utf-8

# In[1]:


import params
# from core import eval_src, eval_tgt, train_src, train_tgt
# from models import Discriminator, LeNetClassifier, LeNetEncoder
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader as DataLoader
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
import utils
from models import LeNetClassifier, LeNetEncoder, Discriminator
from core import pretrain, adapt, test


# In[2]:


source_train_loader, source_test_loader = utils.load_data("MNIST")
target_train_loader, target_test_loader = utils.load_data("USPS")
batch_size = 256


# In[3]:


images, labels = next(iter(target_test_loader))
images.size()


# In[4]:


DEVICE = 'cuda:1'
#Load Models
src_encoder = LeNetEncoder().to(DEVICE)
src_classifier = LeNetClassifier().to(DEVICE)

tgt_encoder = LeNetEncoder().to(DEVICE)
discriminator = Discriminator(input_dims=500, hidden_dims=500, output_dims=2).to(DEVICE)


# In[5]:


#Print models for source
print(src_encoder)
print(src_classifier)


# In[6]:


try:
    src_encoder.load_state_dict(torch.load('src_encoder.pth'))
    src_classifier.load_state_dict(torch.load('src_classifier.pth'))
except FileNotFoundError:
    pretrain.train_src(src_encoder,src_classifier,source_train_loader,DEVICE)


# In[7]:


#Evaluate pretrained model
pretrain.eval_src(src_encoder,src_classifier,source_train_loader,DEVICE)


# In[8]:


#Evaluate pretrained model
pretrain.eval_src(src_encoder,src_classifier,target_test_loader,DEVICE)


# In[9]:


#Load target encoder with source encoder weights
tgt_encoder.load_state_dict(torch.load('src_encoder.pth'))


# In[10]:


#Print models for target and discriminator
print(tgt_encoder)
print(discriminator)


# In[12]:


#Train target encoder using GAN
try:
    discriminator.load_state_dict(torch.load('ADDA-critic-final(MNIST->USPS).pth'))
    tgt_encoder.load_state_dict(torch.load('ADDA-target-encoder-final(MNIST->USPS).pth'))
except FileNotFoundError:
    tgt_encoder, d_losses, g_losses = adapt.train_tgt(src_encoder, tgt_encoder, discriminator, source_train_loader,
                                  target_train_loader,src_classifier, DEVICE)


# In[13]:


#Evaluate pretrained model
pretrain.eval_src(tgt_encoder,src_classifier,target_test_loader,DEVICE)


# In[14]:


#Discriminator Loss
a = plt.plot(d_losses)
plt.title("Discriminator Loss")


# In[15]:


#GAN Loss
a = plt.plot(d_losses)
plt.title("GAN Loss")

