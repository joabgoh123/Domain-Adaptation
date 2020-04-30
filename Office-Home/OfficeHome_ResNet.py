#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader as DataLoader
import matplotlib.pyplot as plt
import os
import torch.nn as nn
import Main
import Models
import torchvision.models as models
import time
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


def load_dataset(root, name, train_percent, mean=(0,0,0), std=(1,1,1)):
    transformations = transforms.Compose([transforms.CenterCrop(224),transforms.RandomHorizontalFlip(), 
                                          transforms.RandomVerticalFlip(), transforms.RandomRotation(30),
                                    transforms.ToTensor(),transforms.Normalize(mean,std)])
    dataset = torchvision.datasets.ImageFolder(os.path.join(root,name),transform=transformations)
    classes_dict = {i:data for i,data in enumerate(dataset.classes)}
    train_length = int(len(dataset) * train_percent)
    test_length = len(dataset) - train_length
    train, test = torch.utils.data.random_split(dataset,[train_length,test_length])
    return train,test, classes_dict


# In[3]:


root = "/raid0/students/student17/OfficeHomeDataset_10072016/"
source_name = "Real World"
target_name = "Product"
source_train, source_test, source_class = load_dataset(root,source_name, 0.8)
target_train, target_test, target_class = load_dataset(root,target_name, 0.8)


# In[4]:


mean_source = (0.5497, 0.5122, 0.4768)
std_source = (0.3168, 0.3110, 0.3199)

mean_target = (0.5497, 0.5122, 0.4768)
std_target = (0.3168, 0.3110, 0.3199)


# In[5]:


source_train, source_test, source_class = load_dataset(root,source_name, 0.9, mean_source, std_source)
target_train, target_test, target_class = load_dataset(root,target_name, 0.8, mean_target, std_target)


# In[6]:


batch_size = 56
num_workers = 8
source_train_loader = torch.utils.data.DataLoader(source_train,batch_size=batch_size, num_workers=num_workers,shuffle=True)
source_test_loader = torch.utils.data.DataLoader(source_test,batch_size=batch_size, num_workers=num_workers,shuffle=True)

target_train_loader = torch.utils.data.DataLoader(target_train,batch_size=batch_size, num_workers=num_workers,shuffle=True)


# In[7]:


class ResNet(nn.Module):
    def __init__(self, num_classes=65):
        super(ResNet, self).__init__()
        self.resnet50 = models.resnet50(pretrained=True)
        self.cls_layer = nn.Linear(1000,num_classes)
        
    def forward(self,x):
        x = self.resnet50(x)
        x = self.cls_layer(x)
        return x


# In[8]:


device = "cuda:0"
model = ResNet()
model.to(device)


# In[9]:


#Define Optimizer
LEARNING_RATE = 0.001
MOMENTUM = 0.9
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD([{'params': model.resnet50.parameters()},
                             {'params': model.cls_layer.parameters(), 'lr':10*LEARNING_RATE}],
                             lr=LEARNING_RATE)


# In[10]:


#Training
epochs = 50
list_epoch_loss = []
for i in range(epochs):
    epoch_loss = 0
    start_time = time.time()
    
    model.train()
    for images, labels in source_train_loader:
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        output = model(images)
        batch_loss = criterion(output,labels)
        
        batch_loss.backward()
        optimizer.step()
        epoch_loss += batch_loss
    
    list_epoch_loss.append(epoch_loss)
    end_time = time.time()
    time_taken = end_time - start_time
    print("Loss :" + str(epoch_loss.item()))
    print("Epoch {:d} completed, time taken: {:f}".format(i+1,time_taken),end="\t")
    
    #Calculate Accuracy
    model.eval()
    num_correct = 0
    for images,labels in source_test_loader:
        images = images.to(device)
        labels = labels.to(device)

        output = model(images)
        pred = torch.argmax(output,1)
        num_correct += torch.sum(pred == labels).item()
    print("Test accuracy: " + str(num_correct/len(source_test_loader.dataset)*100 ) + "%")
    
    #Calculate Accuracy
    model.eval()
    num_correct = 0
    for images,labels in target_train_loader:
        images = images.to(device)
        labels = labels.to(device)

        output = model(images)
        pred = torch.argmax(output,1)
        num_correct += torch.sum(pred == labels).item()
    print("Test accuracy: " + str(num_correct/len(target_train_loader.dataset)*100 ) + "%")


# In[21]:


#Calculate Accuracy
model.eval()
num_correct = 0
for images,labels in source_test_loader:
    images = images.to(device)
    labels = labels.to(device)

    output = model(images)
    pred = torch.argmax(output,1)
    num_correct += torch.sum(pred == labels).item()
print("Test accuracy: " + str(num_correct/len(source_test_loader.dataset)*100 ) + "%")


# In[22]:


#Calculate Accuracy
model.eval()
num_correct = 0
for images,labels in target_train_loader:
    images = images.to(device)
    labels = labels.to(device)

    output = model(images)
    pred = torch.argmax(output,1)
    num_correct += torch.sum(pred == labels).item()
print("Test accuracy: " + str(num_correct/len(target_train_loader.dataset)*100 ) + "%")


# In[ ]:




