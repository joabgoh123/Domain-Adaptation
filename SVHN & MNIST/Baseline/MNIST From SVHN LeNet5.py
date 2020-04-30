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
from collections import OrderedDict
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


device = 'cuda:0'


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


# In[5]:


class LeNet5(nn.Module):
    """
    Input - 1x32x32
    C1 - 6@28x28 (5x5 kernel)
    tanh
    S2 - 6@14x14 (2x2 kernel, stride 2) Subsampling
    C3 - 16@10x10 (5x5 kernel, complicated shit)
    tanh
    S4 - 16@5x5 (2x2 kernel, stride 2) Subsampling
    C5 - 120@1x1 (5x5 kernel)
    F6 - 84
    tanh
    F7 - 10 (Output)
    """
    def __init__(self):
        super(LeNet5, self).__init__()
        self.convnet = nn.Sequential(OrderedDict([
            ('c1', nn.Conv2d(3, 6, kernel_size=(5, 5))),
            ('relu1', nn.ReLU()),
            ('s2', nn.MaxPool2d(kernel_size=(2, 2), stride=2)),
            ('c3', nn.Conv2d(6, 16, kernel_size=(5, 5))),
            ('relu3', nn.ReLU()),
            ('s4', nn.MaxPool2d(kernel_size=(2, 2), stride=2)),
            ('c5', nn.Conv2d(16, 120, kernel_size=(5, 5))),
            ('relu5', nn.ReLU())
        ]))

        self.fc = nn.Sequential(OrderedDict([
            ('f6', nn.Linear(120, 84)),
            ('relu6', nn.ReLU()),
        ]))

        self.fc_cls = nn.Sequential(OrderedDict([
            ('f7', nn.Linear(84, 10))
        ]))
    
    def forward(self, x):
        x = self.convnet(x)
        x = x.view(-1,120)
        x = self.fc(x)
        x = self.fc_cls(x)
        return x
    
model = LeNet5()
model.to(device)


# In[6]:


#Load trained SVHN Weights
model = LeNet5()
model.load_state_dict(torch.load("SVHN_LeNet5_10epochs.pth"))
model.to(device)
model.eval()


# In[7]:


#Function to print test accuracy
def compute_accuracy():
    num_correct = 0
    for images,labels in test_loader:
        model.eval()
        images = images.to(device)
        labels = labels.to(device)
        images = torch.cat((images,images,images),dim=1)
        images = nn.functional.pad(images,(2,2,2,2))
        output = model(images)
        pred = torch.argmax(output,1)
        num_correct += torch.sum(pred == labels).item()
    accuracy = (num_correct / len(test_data)) * 100
    print("Accuracy is : {:f}%".format(accuracy))


# In[8]:


#SVHN -> MNIST
compute_accuracy()


# In[9]:


#Freeze all layers
for param in model.parameters():
    param.requires_grad = False
#Set final FC layer to trainable
for param in model.fc_cls.parameters():
    param.requires_grad = True


# In[10]:


#Define Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=0.001)


# In[11]:


#Training
losses = []
epochs = 10
model.train()
for i in range(epochs):
    epoch_loss = 0
    start_time = time.time()
    for images,labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)
        images = torch.cat((images,images,images),dim=1)
        images = nn.functional.pad(images,(2,2,2,2))
        
        #Forward pass
        output = model(images)
        batch_loss = criterion(output,labels)
        
        #Backward pass
        batch_loss.backward()
        optimizer.step()
        epoch_loss += batch_loss.item()
    losses.append(epoch_loss)
    end_time = time.time()
    time_taken = end_time - start_time
    print("Epoch {:d} completed, time taken: {:f}".format(i+1,time_taken))
        


# In[12]:


compute_accuracy()


# In[13]:


mean = (0,0,0)
std = (1,1,1)


# In[14]:


#LOAD SVHN DATASET

device = 'cuda:0'
#Transforms
transform = transforms.Compose([transforms.ToTensor(),
                               transforms.Normalize(mean,std)])

#Download Datasets
train_data = datasets.SVHN('~\\Datasets', download=True,transform=transform)
test_data = datasets.SVHN('~\\Datasets',download=True, split='test', transform=transform)

#Split train data to train and val
train_length = int(len(train_data)*0.8)
val_length = len(train_data) - train_length
train_data, val_data = torch.utils.data.random_split(train_data,[train_length,val_length])

#Data Loaders
train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=True)

#Check data Sizes
print("Number of Training Examples : {:d} ".format(len(train_data)))
print("Number of Validation Examples : {:d} ".format(len(val_data)))
print("Number of Test Examples : {:d} ".format(len(test_data)))
print("Shape of data: " + str(train_data[0][0].size()))

#Data Loaders
train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=True)


# In[15]:


get_ipython().run_cell_magic('time', '', '#Calculate Accuracy\nmodel.eval()\nnum_correct = 0\nfor images,labels in test_loader:\n    images = images.to(device)\n    labels = labels.to(device)\n    output = model(images)\n    pred = torch.argmax(output,1)\n    num_correct += torch.sum(pred == labels).item()\nprint("Test accuracy: " + str(num_correct/len(test_loader.dataset)*100 ) + "%")')


# In[ ]:




