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
from collections import OrderedDict
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


mean = (0,0,0)
std = (1,1,1)


# In[10]:


device = "cuda:1"
transform = transforms.Compose([transforms.ToTensor()])
#Download Datasets
train_data = datasets.MNIST('~\\Datasets',download=True, train=True, transform=transform)
test_data = datasets.MNIST('~\\Datasets',download=False, train=False, transform=transform)

#Check data Sizes
print("Number of Training Examples : {:d} ".format(len(train_data)))
print("Number of Test Examples : {:d} ".format(len(test_data)))
print("Shape of data: " + str(train_data[0][0].size()))

train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=True)


# In[4]:


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


# In[5]:


#Define Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=0.001)


# In[6]:


#Record losses
losses = []
#Training
epochs = 10
for i in range(epochs):
    epoch_loss = 0
    val_loss = 0
    val_correct = 0
    start_time = time.time()
    model.train()
    for images,labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)
        images = torch.cat((images,images,images),dim=1)
        images = nn.functional.pad(images,(2,2,2,2))
        optimizer.zero_grad()
        
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
    print("Epoch {:d} completed, time taken: {:f}".format(i+1,time_taken),end="\t")
    print("Training Loss: " + str(epoch_loss))
 
    


# In[11]:


get_ipython().run_cell_magic('time', '', '#Calculate Accuracy\nmodel.eval()\nnum_correct = 0\nfor images,labels in test_loader:\n    images = images.to(device)\n    labels = labels.to(device)\n    images = torch.cat((images,images,images),dim=1)\n    images = nn.functional.pad(images,(2,2,2,2))\n    output = model(images)\n    pred = torch.argmax(output,1)\n    num_correct += torch.sum(pred == labels).item()\nprint("Test accuracy: " + str(num_correct/len(test_loader.dataset)*100 ) + "%")')


# In[8]:


#Plot losses
plt.plot(losses, 'orange')
plt.xticks(np.arange(1, epochs))
plt.title("MNIST LeNet losses")
plt.xlabel("Epochs")
plt.ylabel("Loss")


# In[9]:


torch.save(model.state_dict(), "MNIST_LeNet5.pth")


# In[ ]:




