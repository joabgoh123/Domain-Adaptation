#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
sys.path.insert(1, '/raid0/students/student17/All Transfer Learning Methods')


# In[2]:


import torch
import math
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets,transforms
import torch.nn as nn
import torch.nn.functional as F
import time
import functools
from torch.autograd import Variable
from image_loader_methods import load_dataloaders, showImage
import image_loader
import DDC_resnet as DDC
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')

torch.multiprocessing.set_sharing_strategy('file_system')


# In[3]:


#Load datasets
source_train_loader, source_test_loader, classes = load_dataloaders("ImageNet_train", batch_size=24, num_workers = 16, train_percent=0.8)
target_train_loader, target_test_loader, classes = load_dataloaders("DarkImage", batch_size=24, num_workers = 16, train_percent=0.8)


# In[4]:


#Show images
images, labels = next(iter(source_train_loader))
showImage(images,labels,title="Images")


# In[5]:


#Specify Device
device = 'cuda:2'
#Load model
model = DDC.DDC()
model.to(device)
# model.load_state_dict(torch.load("ddc_resnet_100e.pth"))


# In[6]:


#Define Optimizer
LEARNING_RATE = 0.001
MOMENTUM = 0.9
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD([{'params': model.convNet.parameters()},
                             {'params': model.fc.parameters(), 'lr':LEARNING_RATE*10}],
                             lr=LEARNING_RATE)


# In[7]:


source, target = list(enumerate(source_train_loader)), list(enumerate(target_train_loader))
train_steps = min(len(source), len(target)) - 1


# In[8]:


list_ddc_loss = []
list_cls_loss = []


# In[ ]:


#Training
epochs = 30
_lambda = 0
for i in range(epochs):
    ddc_loss_epoch = 0
    class_loss = 0
#     _lambda = 2 / (1 + math.exp(-10 * (i) / epochs)) - 1
    _lambda += 0.0001
    epoch_loss = 0
    start_time = time.time()
    model.train()
    for batch_idx in range(train_steps):
        _, (source_data, source_label) = source[batch_idx]
        _, (target_data, _) = target[batch_idx]
        source_data = source_data.to(device)
        source_label = source_label.to(device)
        target_data = target_data.to(device)
        
        
        optimizer.zero_grad()
        #Forward pass
        out_source, out_target, ddc_loss = model(source_data,target_data)
        #Calculate loss
        classification_loss = criterion(out_source,source_label)
        total_loss = classification_loss + (_lambda * ddc_loss)
        class_loss +=classification_loss

        #Backward pass
        total_loss.backward()
        optimizer.step()
        ddc_loss_epoch += ddc_loss.item()
        epoch_loss += total_loss.item()
    end_time = time.time()
    time_taken = end_time - start_time
    print("ddc loss " + str(ddc_loss_epoch))
    print("Epoch {:d} completed, time taken: {:f}".format(i+1,time_taken),end="\t")
    print("Training Loss: " + str(epoch_loss))
    list_ddc_loss.append(ddc_loss_epoch)
    list_cls_loss.append(class_loss)
    num_correct = 0
    for images,labels in source_test_loader:
        images = images.to(device)
        labels = labels.to(device)

        output,_ ,_  = model(images,images)
        pred = torch.argmax(output,1)
        num_correct += torch.sum(pred == labels).item()
    print("Test accuracy: " + str(num_correct/len(source_test_loader.dataset)*100 ) + "%")
    num_correct = 0
    for images,labels in target_test_loader:
        images = images.to(device)
        labels = labels.to(device)
        output,_ ,_  = model(images,images)
        pred = torch.argmax(output,1)
        num_correct += torch.sum(pred == labels).item()
    print("Test accuracy: " + str(num_correct/len(target_test_loader.dataset)*100 ) + "%")
     #Validation
#     model.eval()
#     for images, labels in val_loader:
#         images = images.to(device)
#         labels = labels.to(device)
        
#         #Forward pass
#         output = model(images)
#         batch_loss = criterion(output,labels)
#         val_loss += batch_loss.item()
#         pred = torch.argmax(output,1)
#         val_correct += torch.sum(pred == labels).item()
        
#     print("Validation Loss: " + str(val_loss),end="\t")
#     print("Accuracy: " + str(val_correct/len(val_data)*100) + "%")
    


# In[ ]:


num_correct = 0 
for images,labels in target_test_loader:
    images = images.to(device)
    labels = labels.to(device)

    output,_ ,_  = model(images,images)
    pred = torch.argmax(output,1)
    num_correct += torch.sum(pred == labels).item()
print("Test accuracy: " + str(num_correct/len(target_test_loader.dataset)*100 ) + "%")


# In[ ]:





# In[11]:


pd.DataFrame(list_ddc_loss).to_csv("ddc_loss.csv")
pd.DataFrame(list_cls_loss).to_csv("cls_loss.csv")


# In[11]:


# torch.save(model.state_dict(), "ddc_resnet_100e_newData.pth")


# In[ ]:




