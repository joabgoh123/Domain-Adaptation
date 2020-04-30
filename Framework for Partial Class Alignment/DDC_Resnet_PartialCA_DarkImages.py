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
from torchvision import datasets,transforms, models
import torch.nn as nn
import torch.nn.functional as F
import time
import functools
from torch.autograd import Variable
from image_loader_methods import load_dataloaders, showImage
import image_loader
import random
import DDC_resnet as DDC
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')

torch.multiprocessing.set_sharing_strategy('file_system')


# In[3]:


target_train_loader, target_test_loader, classes = load_dataloaders("DarkImage", batch_size=9999999, num_workers = 8, train_percent=0.8)
darknet_images, darknet_labels = next(iter(target_train_loader))


# In[4]:


#Load datasets
source_train_loader, source_test_loader, classes = load_dataloaders("ImageNet_train", batch_size=24, num_workers = 8, train_percent=0.8)
target_train_loader, target_test_loader, classes = load_dataloaders("DarkImage", batch_size=24, num_workers = 8, train_percent=0.8)


# In[5]:


# imagenet_images = torch.load('/raid0/students/student17/All Transfer Learning Methods/imagenet_images_v2.pt')
# imagenet_labels = torch.load('/raid0/students/student17/All Transfer Learning Methods/imagenet_labels_v2.pt')
# darknet_images = torch.load('/raid0/students/student17/All Transfer Learning Methods/darkdata_images_v2.pt')
# darknet_labels = torch.load('/raid0/students/student17/All Transfer Learning Methods/darkdata_labels_v2.pt')


# In[6]:


# class ResNet(nn.Module):
#     def __init__(self, num_classes=11):
#         super(ResNet, self).__init__()
#         self.resnet50 = models.resnet50(pretrained=True)
#         self.cls_layer = nn.Linear(1000,num_classes)
        
#     def forward(self,x):
#         x = self.resnet50(x)
#         x = self.cls_layer(x)
#         return x


# In[7]:


# device = "cuda:0"
# model = ResNet()
# model.to(device)
# model.load_state_dict(torch.load("ResNet50_NoTransferLearning.pth"))


# In[8]:


# #Specify Device
# device = 'cuda:0'
# #Load model
# model = DDC.DDC()
# model.to(device)
# model.load_state_dict(torch.load("ddc_resnet_100e_newData.pth"))


# In[9]:


#Specify Device
device = 'cuda:1'
#Load model
model = DDC.DDC()
model.to(device)
model.load_state_dict(torch.load("ddc_resnet_100e_newData.pth"))


# In[10]:


num_correct = 0
for images,labels in target_test_loader:
    model.eval()
    images = images.to(device)
    labels = labels.to(device)
    output, _,_  = model(images,images)
    pred = torch.argmax(output,1)
    num_correct += torch.sum(pred == labels).item()
print("Test accuracy: " + str(num_correct/len(target_test_loader.dataset)*100 ) + "%")


# In[11]:


tensors = [[] for i in range(11)]
for i in range(len(darknet_images)):
    print(i)
    model.eval()
    img = darknet_images[i].unsqueeze(dim=0)
    img = img.to(device)
    output, _,_ = model(img,img)
    pred = torch.argmax(output,1).item()
    tensors[pred].append(i)


# In[12]:


#Specify Device
device = 'cuda:1'
#Load model
model = DDC.DDC()
model.to(device)


# In[13]:


#Define Optimizer
LEARNING_RATE = 0.001
MOMENTUM = 0.9
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD([{'params': model.convNet.parameters()},
                             {'params': model.fc.parameters(), 'lr':LEARNING_RATE*10}],
                             lr=LEARNING_RATE)


# In[14]:


def retrieve_idx(labels):
    img_indexs = []
    for label in labels.tolist():
        img_indexs.append(random.choice(tensors[label]))
    return img_indexs


# In[15]:


list_ddc_loss = []
list_cls_loss = []


# In[16]:


max_accuracy = 0


# In[ ]:


#Training
epochs = 30
_lambda = 0
max_accuracy = 0
for i in range(1,epochs):
    ddc_loss_epoch = 0
    class_loss = 0
#     _lambda = 2 / (1 + math.exp(-10 * (i) / epochs)) - 1
    _lambda += 0.0001
    epoch_loss = 0
    start_time = time.time()
    model.train()
    for images,labels in source_train_loader:
        source_data = images.to(device)
        source_labels = labels.to(device)
        #target_idx = retrieve_idx(source_labels)
        random_ints = np.random.choice(len(darknet_images),source_data.size()[0], replace=False)
        target_data = darknet_images[random_ints].to(device)

        
        optimizer.zero_grad()
        #Forward pass
        out_source, out_target, ddc_loss = model(source_data,target_data)
        #Calculate loss
        classification_loss = criterion(out_source,source_labels)
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
        model.eval()
        images = images.to(device)
        labels = labels.to(device)
        output,_ ,_  = model(images,images)
        pred = torch.argmax(output,1)
        num_correct += torch.sum(pred == labels).item()
    print("Test accuracy: " + str(num_correct/len(target_test_loader.dataset)*100 ) + "%")


    
    accuracy = num_correct/len(target_test_loader.dataset)
    if accuracy > max_accuracy:
        torch.save(model.state_dict(), "ddc_resnet_100e_weakclassifier_newData_PrevDDC.pth")
        max_accuracy = accuracy
        print("Model updated")
    
#     if i % 10 == 0:
#         tensors = [[] for i in range(11)]
#         for i in range(len(darknet_images)):
#             print(i)
#             model.eval()
#             img = darknet_images[i].unsqueeze(dim=0)
#             img = img.to(device)
#             output,_,_ = model(img,img)
#             pred = torch.argmax(output,1).item()
#             tensors[pred].append(i)

    


# In[ ]:


tensors = [[] for i in range(11)]
for i in range(len(darknet_images)):
    print(i)
    model.eval()
    img = darknet_images[i].unsqueeze(dim=0)
    img = img.to(device)
    output,_,_ = model(img,img)
    pred = torch.argmax(output,1).item()
    tensors[pred].append(i)


# In[ ]:


# torch.save(model.state_dict(), "ddc_resnet_100e_weakclassifier_newData_selfAdjusted.pth")


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


# pd.DataFrame(list_ddc_loss).to_csv("ddc_loss_weakclassifier_newData.csv")
# pd.DataFrame(list_cls_loss).to_csv("cls_loss.csv_weakclassifier_newData.csv")


# In[ ]:


# torch.save(model.state_dict(), "ddc_resnet_100e_weakclassifier_newData.pth")


# In[ ]:




