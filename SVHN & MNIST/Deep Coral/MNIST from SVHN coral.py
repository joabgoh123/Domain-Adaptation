#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets,transforms
import torch.nn as nn
import torch.nn.functional as F
import time
import functools
import utils
import coral
from torch.autograd import Variable
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


device = 'cuda:1'


# In[3]:


transform = transforms.Compose([transforms.ToTensor()])
#Download Datasets
train_data = datasets.MNIST('~\\Datasets',download=True, train=True, transform=transform)
test_data = datasets.MNIST('~\\Datasets',download=True, train=False, transform=transform)

#Check data Sizes
print("Number of Training Examples : {:d} ".format(len(train_data)))
print("Number of Test Examples : {:d} ".format(len(test_data)))
print("Shape of data: " + str(train_data[0][0].size()))

train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=True)


# In[4]:


#Function to display images
def showImage(images,labels,title="SVHN Dataset"):
    rows = 2
    cols = 5
    k = 0
    if type(labels) == torch.Tensor:
        labels = labels.cpu()
        labels = np.array(labels).tolist()
    img_to_display = images[:10].permute(0,2,3,1).cpu()
    labels = labels[:10]
    fig, ax = plt.subplots(rows,cols,figsize=(10, 5))
    for i in range(rows):
        for j in range(cols):
            ax[i][j].imshow(img_to_display[k])
            _ = ax[i][j].set(xlabel = labels[k])
            k+=1
    _ = fig.suptitle(title)


# In[17]:


#Load model
model = coral.DeepCORAL()
model.to(device)

#Load pretrained
# pretrain_dict = torch.load("C:\\Users\\Joab-PC\\Desktop\\Personal Documents\\Jupyter Notebooks\\svhn_model_statedict.pth")
# model.convNet.state_dict().update(pretrain_dict)
model.load_state_dict(torch.load('svhn_coral.pth'))


# In[18]:


num_correct = 0
for images,labels in test_loader:
    images = images.to(device)
    labels = labels.to(device)
    images = torch.cat((images,images,images),dim=1)
    images = nn.functional.pad(images,(2,2,2,2))
    output, _ = model(images, images)
    pred = torch.argmax(output,1)
    num_correct += torch.sum(pred == labels).item()
accuracy = (num_correct / len(test_data)) * 100
print("Accuracy is : {:f}%".format(accuracy))




