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
from torch.autograd import Variable
from image_loader_methods import load_dataloaders, showImage
from torchvision import models
import image_loader
get_ipython().run_line_magic('matplotlib', 'inline')

torch.multiprocessing.set_sharing_strategy('file_system')


# In[3]:


#Load datasets
source_train_loader, source_test_loader, classes = load_dataloaders("ImageNet_train", batch_size=48, num_workers = 4, train_percent=0.9)
target_train_loader, target_test_loader, classes = load_dataloaders("DarkImage", batch_size=48, num_workers = 4,train_percent=0.99)


# In[4]:


#Show images
images, labels = next(iter(source_train_loader))
showImage(images,labels,title="Images")


# In[5]:


resnet50 = models.resnet50(pretrained=True)


# In[6]:


class ResNet(nn.Module):
    def __init__(self, num_classes=11):
        super(ResNet, self).__init__()
        self.resnet50 = models.resnet50(pretrained=True)
        self.cls_layer = nn.Linear(1000,num_classes)
        
    def forward(self,x):
        x = self.resnet50(x)
        x = self.cls_layer(x)
        return x


# In[9]:


device = "cuda:0"
model = ResNet()
model.to(device)
model.load_state_dict(torch.load("ResNet50_NoTransferLearning.pth"))


# In[10]:


#Define Optimizer
LEARNING_RATE = 0.001
MOMENTUM = 0.9
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD([{'params': model.resnet50.parameters()},
                             {'params': model.cls_layer.parameters(), 'lr':10*LEARNING_RATE}],
                             lr=LEARNING_RATE)


# In[11]:


#Training
epochs = 10
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
    
    list_epoch_loss.append(epoch_loss.item())
    end_time = time.time()
    time_taken = end_time - start_time
    print("Loss :" + str(epoch_loss.item()))
    print("Epoch {:d} completed, time taken: {:f}".format(i+1,time_taken),end="\t")


# In[12]:


# list_epoch_loss = [112.5131,74.5908,61.3703,52.2725,47.4424,40.0220,33.7726,31.9198,27.8154,25.5845]
plt.plot(list_epoch_loss, 'orange')
plt.xticks(np.arange(1, epochs))
plt.title("ResNet50 No Transfer Learning")
plt.xlabel("Epochs")
plt.ylabel("Loss")


# In[13]:


torch.save(model.state_dict(), "ResNet50_NoTransferLearning.pth")


# In[14]:


#Calculate Accuracy
num_correct = 0
for images,labels in source_test_loader:
    model.eval()
    images = images.to(device)
    labels = labels.to(device)

    output = model(images)
    pred = torch.argmax(output,1)
    num_correct += torch.sum(pred == labels).item()
print("Test accuracy: " + str(num_correct/len(source_test_loader.dataset)*100 ) + "%")


# In[17]:


all_labels = []
all_pred = []


# In[19]:


#Calculate Accuracy
num_correct = 0
for images,labels in target_train_loader:
    model.eval()
    images = images.to(device)
    labels = labels.to(device)

    all_labels += labels.tolist()
    all_pred += pred.tolist()
    
    output = model(images)
    pred = torch.argmax(output,1)
    num_correct += torch.sum(pred == labels).item()
print("Test accuracy: " + str(num_correct/len(target_train_loader.dataset)*100 ) + "%")


# In[20]:


accuracies = []
for i in range(11):
    accuracies.append(all_pred.count(i) /all_labels.count(i))


# In[21]:


accuracies


# In[11]:


num_correct = 0
for images, labels in target_train_loader:
    model.eval()
    images, labels = images.to(device), labels.to(device)
    output_class = model(images)
    pred = torch.argmax(output_class,1)
    
    ground_truth = np.argwhere(labels.cpu()==2)
    prediction = np.argwhere(pred.cpu()== 2)
    
    list_ground_truth = ground_truth.squeeze().tolist()
    list_prediction = prediction.squeeze().tolist()
    try:
        for element in list_ground_truth:
            if element in list_prediction:
                num_correct+=1
    except:
        num_correct+=1
    
print(num_correct)
accuracy = (num_correct / len(target_train_loader.dataset)) * 100
print("Accuracy is : {:f}%".format(accuracy))


# In[ ]:




