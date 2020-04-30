#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
sys.path.insert(1, '/raid0/students/student17/All Transfer Learning Methods')


# In[2]:


import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets,transforms
import torch.nn as nn
import torch.nn.functional as F
import time
import functools
import utils
import coral_resnet
from torch.autograd import Variable
from image_loader_methods import load_dataloaders, showImage
import image_loader
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')

torch.multiprocessing.set_sharing_strategy('file_system')


# In[3]:


#Load datasets
source_train_loader, source_test_loader, classes = load_dataloaders("ImageNet_train", batch_size=24, num_workers = 16)
target_train_loader, target_test_loader, classes = load_dataloaders("DarkImage", batch_size=24, num_workers = 16,train_percent=0.8)


# In[4]:


#Show images
images, labels = next(iter(source_train_loader))
showImage(images,labels,title="Images")


# In[5]:


source, target = list(enumerate(source_train_loader)), list(enumerate(target_train_loader))
train_steps = min(len(source), len(target))


# In[6]:


device = "cuda:2"
model = coral_resnet.DeepCORAL()
# if torch.cuda.device_count() > 1:
#     print("Let's use", torch.cuda.device_count(), "GPUs!")
#     # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
#     model = nn.DataParallel(model)

model.to(device)
# model.load_state_dict(torch.load("coral_resnet.pth"))


# In[7]:


model.load_state_dict(torch.load('coral_resnet_100e.pth'))


# In[7]:


#Define Optimizer, SINGLE GPU
LEARNING_RATE = 0.001
MOMENTUM = 0.9
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD([{'params': model.convNet.parameters()},
                             {'params': model.fc.parameters(), 'lr':10*LEARNING_RATE}],
                             lr=LEARNING_RATE)


# In[8]:


# #Define Optimizer, MULTI GPU
# LEARNING_RATE = 0.001
# MOMENTUM = 0.9
# criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam([{'params': model.module.convNet.parameters()},
#                              {'params': model.module.fc.parameters(), 'lr':10*LEARNING_RATE}],
#                              lr=LEARNING_RATE)


# In[ ]:


#Training
list_class_loss = []
list_coral_loss = []
epochs = 30
_lambda = 0.08
for i in range(epochs):
    coral_epoch = 0
    class_loss = 0
    _lambda+= 0.004
    epoch_loss = 0
    start_time = time.time()
    model.train()
    for batch_idx in range(train_steps):
        _, (source_data, source_label) = source[batch_idx]
        _, (target_data, _) = target[batch_idx]
        
        source_data, source_label = Variable(source_data), Variable(source_label)
        target_data = Variable(target_data)
        
        source_data = source_data.to(device)
        source_label = source_label.to(device)
        target_data = target_data.to(device)
        
        
        
        optimizer.zero_grad()
        #Forward pass
        out_source, out_target = model(source_data,target_data)
        #Calculate loss
        classification_loss =  torch.nn.functional.cross_entropy(out_source, source_label)
#         classification_loss = criterion(out_source,source_label)
        coral_loss = coral_resnet.coral_loss(out_source,out_target)
        total_loss = classification_loss + (_lambda * coral_loss)

        #Backward pass
        total_loss.backward()
        optimizer.step()
        coral_epoch += coral_loss.item()
        class_loss +=classification_loss.item()
        epoch_loss += total_loss.item()
    end_time = time.time()
    time_taken = end_time - start_time
    print("coral loss " + str(coral_epoch))
    print("class loss " + str(class_loss))
    print("Epoch {:d} completed, time taken: {:f}".format(i+1,time_taken),end="\t")
    print("Training Loss: " + str(epoch_loss))
    list_class_loss.append(class_loss)
    list_coral_loss.append(coral_epoch)
    
    #Calculate Accuracy
    model.eval()
    num_correct = 0
    for images,labels in source_test_loader:
        model.eval()
        images = images.to(device)
        labels = labels.to(device)

        output,_ = model(images,images)
        pred = torch.argmax(output,1)
        num_correct += torch.sum(pred == labels).item()
    print("Test accuracy: " + str(num_correct/len(source_test_loader.dataset)*100 ) + "%")
    
    num_correct = 0
    for images,labels in target_test_loader:
        model.eval()
        images = images.to(device)
        labels = labels.to(device)

        output,_ = model(images,images)
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


# In[19]:


pd.DataFrame(list_class_loss).to_csv('class_loss.csv')
pd.DataFrame(list_coral_loss).to_csv('coral_loss.csv')


# In[21]:


# #Multi GPU
# torch.save(model.module.state_dict(), "coral_resnet.pth")
# Single GPU
torch.save(model.state_dict(), "coral_resnet_100e.pth")


# In[30]:


all_labels = []
all_pred = []


# In[41]:


get_ipython().run_cell_magic('time', '', '#Calculate Accuracy\nnum_correct = 0\nfor images,labels in target_train_loader:\n    model.eval()\n    images = images.to(device)\n    labels = labels.to(device)\n    \n    output,_ = model(images,images)\n    pred = torch.argmax(output,1)\n    \n    all_labels += labels.tolist()\n    all_pred += pred.tolist()\n    \n    num_correct += torch.sum(pred == labels).item()\nprint("Test accuracy: " + str(num_correct/len(target_train_loader.dataset)*100 ) + "%")')


# In[22]:


all_labels += labels.tolist()


# In[32]:


all_labels.count(1)


# In[38]:


accuracies = []
for i in range(11):
    accuracies.append(all_pred.count(i) /all_labels.count(i))


# In[39]:


accuracies


# In[40]:


classes[4]

