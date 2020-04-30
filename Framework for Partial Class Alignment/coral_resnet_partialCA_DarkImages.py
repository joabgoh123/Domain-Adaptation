#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
sys.path.insert(1, '/raid0/students/student17/All Transfer Learning Methods')


# In[2]:


import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets,transforms, models
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
import random
get_ipython().run_line_magic('matplotlib', 'inline')

torch.multiprocessing.set_sharing_strategy('file_system')


# In[3]:


#Load datasets
source_train_loader, source_test_loader, classes = load_dataloaders("ImageNet_train", batch_size=24, num_workers = 16)
target_train_loader, target_test_loader, classes = load_dataloaders("DarkImage", batch_size=24, num_workers = 16,train_percent=0.8)


# In[4]:


imagenet_images = torch.load('/raid0/students/student17/All Transfer Learning Methods/imagenet_images_v2.pt')
imagenet_labels = torch.load('/raid0/students/student17/All Transfer Learning Methods/imagenet_labels_v2.pt')
darknet_images = torch.load('/raid0/students/student17/All Transfer Learning Methods/darkdata_images_v2.pt')
darknet_labels = torch.load('/raid0/students/student17/All Transfer Learning Methods/darkdata_labels_v2.pt')


# In[5]:


device = "cuda:0"
model = coral_resnet.DeepCORAL()
model.to(device)
model.load_state_dict(torch.load("ddc_resnet_100e_newData-Copy1.pth"))


# In[10]:


get_ipython().run_cell_magic('time', '', '#Calculate Accuracy\nnum_correct = 0\nfor images,labels in source_test_loader:\n    model.eval()\n    images = images.to(device)\n    labels = labels.to(device)\n    output, _ = model(images,images)\n    pred = torch.argmax(output,1)\n    num_correct += torch.sum(pred == labels).item()\nprint("Test accuracy: " + str(num_correct/len(source_test_loader.dataset)*100 ) + "%")')


# In[12]:


get_ipython().run_cell_magic('time', '', '#Calculate Accuracy\nnum_correct = 0\nfor images,labels in target_test_loader:\n    model.eval()\n    images = images.to(device)\n    labels = labels.to(device)\n    \n    output, _ = model(images, images)\n    pred = torch.argmax(output,1)\n    num_correct += torch.sum(pred == labels).item()\nprint("Test accuracy: " + str(num_correct/len(target_test_loader.dataset)*100 ) + "%")')


# In[13]:


tensors = [[] for i in range(11)]
for i in range(len(darknet_images)):
    model.eval()
    img = darknet_images[i].unsqueeze(dim=0)
    img = img.to(device)
    output, _ = model(img, img)
    pred = torch.argmax(output,1).item()
    tensors[pred].append(i)
    print(i)


# In[2]:


tensors


# In[15]:


#Define Optimizer, SINGLE GPU
LEARNING_RATE = 0.001
MOMENTUM = 0.9
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD([{'params': model.convNet.parameters()},
                             {'params': model.fc.parameters(), 'lr':10*LEARNING_RATE}],
                             lr=LEARNING_RATE)


# In[16]:


def retrieve_idx(labels):
    img_indexs = []
    for label in labels.tolist():
        img_indexs.append(random.choice(tensors[label]))
    return img_indexs


# In[19]:


#Training
list_class_loss = []
list_coral_loss = []
epochs = 10
_lambda = 0.01
for i in range(1,epochs):
    coral_epoch = 0
    class_loss = 0
#     _lambda+= 0.004
    epoch_loss = 0
    start_time = time.time()
    model.train()
    prev_j = 0
    for j in range(20,len(imagenet_images),20):
        random_ints = np.random.choice(len(imagenet_images),20, replace=False)
        source_images = imagenet_images[random_ints].to(device)
        source_labels = imagenet_labels[random_ints].to(device)
        target_idx = retrieve_idx(source_labels)
        target_images = darknet_images[target_idx].to(device)

        optimizer.zero_grad()
        #Forward pass
        out_source, out_target = model(source_images, target_images)
        
        pred = torch.argmax(out_source,1)
        #Calculate loss
        classification_loss =  criterion(out_source, source_labels)
    #         classification_loss = criterion(out_source,source_label)
        coral_loss = coral_resnet.coral_loss(out_source, out_target)
        total_loss = classification_loss + (_lambda * coral_loss)
        prev_j = j

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
    torch.save(model.state_dict(), "coral_resnet_100e_weakClassifier_2.pth")
    
#     if i % 10 == 0:
#         tensors = [[] for i in range(11)]
#         for i in range(len(darknet_images)):
#             img = darknet_images[i].unsqueeze(dim=0)
#             img = img.to(device)
#             output,_ = model(img,img)
#             pred = torch.argmax(output,1).item()
#             tensors[pred].append(i)
        
    #Calculate Accuracy
    
    num_correct = 0
    for images,labels in target_test_loader:
        model.eval()
        images = images.to(device)
        labels = labels.to(device)

        output,_ = model(images,images)
        pred = torch.argmax(output,1)
        num_correct += torch.sum(pred == labels).item()
    print("Test accuracy: " + str(num_correct/len(target_test_loader.dataset)*100 ) + "%")


# In[ ]:


tensors = [[] for i in range(11)]
for i in range(len(darknet_images)):
    img = darknet_images[i].unsqueeze(dim=0)
    img = img.to(device)
    output, _ = model(img,img)
    pred = torch.argmax(output,1).item()
    tensors[pred].append(i)
    print(i)


# In[19]:


pd.DataFrame(list_class_loss).to_csv('class_loss_weakClassifier.csv')
pd.DataFrame(list_coral_loss).to_csv('coral_loss_weakClassifier.csv')


# In[21]:


# Single GPU
torch.save(model.state_dict(), "coral_resnet_100e_weakClassifier.pth")

