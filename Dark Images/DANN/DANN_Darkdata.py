#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
sys.path.insert(1, '/raid0/students/student17/All Transfer Learning Methods')


# In[2]:


import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader as DataLoader
import matplotlib.pyplot as plt
import os
import pandas as pd
import torch.nn as nn
import Main
import Models
import numpy as np
from image_loader_methods import load_dataloaders, showImage
import torchvision.models as models
get_ipython().run_line_magic('matplotlib', 'inline')

torch.multiprocessing.set_sharing_strategy('file_system')


# In[3]:


#Load datasets
source_train_loader, source_test_loader, classes = load_dataloaders("ImageNet_train", batch_size=36, num_workers = 16)
target_train_loader, target_test_loader, classes = load_dataloaders("DarkImage", batch_size=36, num_workers = 16, train_percent=0.99)


# In[4]:


DEVICE = "cuda:2"
model = Models.ResNet()
# model.load_state_dict(torch.load("C:\\Users\\Joab-PC\\Desktop\\Personal Documents\\Jupyter Notebooks\\DANN_model_RealWorld.pth"))
# model.load_state_dict(torch.load("DANN_model_RealWorld_SGD3.pth"))
model.to(DEVICE)

optimizer = torch.optim.SGD([
        {"params": model.resnet50_features.parameters(), "lr": 0.001},
        {"params": model.class_classifier.parameters(), "lr": 0.01},
        {"params": model.domain_classifier.parameters(), "lr": 0.01},
    ],lr=0.01)


# In[5]:


model.load_state_dict(torch.load('DANN_ResNet_Darkdata_100e.pth'))


# In[6]:


batch_size = 36
train_acc_ , target_acc_, train_lost, s_domain_loss, t_domain_loss = Main.train(model,optimizer,source_train_loader,target_train_loader,1,2,DEVICE,batch_size)


# In[15]:


pd.DataFrame(train_acc_).to_csv('train_acc')
pd.DataFrame(target_acc_).to_csv('target_acc.csv')
pd.DataFrame(train_lost).to_csv('train_loss.csv')
pd.DataFrame(s_domain_loss).to_csv('s_domain_loss.csv')
pd.DataFrame(t_domain_loss).to_csv('t_domain_loss.csv')


# In[ ]:


torch.save(model.state_dict(), "DANN_ResNet_Darkdata_100e.pth")


# In[ ]:


images, labels = next(iter(target_train_loader))
images, labels = images.to(DEVICE), labels.to(DEVICE)
output_class, output_domain = model(images,0.1)


# In[ ]:


num_correct = 0
pred = torch.argmax(output_class,1)
num_correct += torch.sum(pred == labels).item()
accuracy = (num_correct / 56) * 100
print("Accuracy is : {:f}%".format(accuracy))


# In[6]:


num_correct = 0
for images, labels in target_test_loader:
    model.eval()
    images, labels = images.to(DEVICE), labels.to(DEVICE)
    output_class, output_domain = model(images,0)
    pred = torch.argmax(output_class,1)
    num_correct += torch.sum(pred == labels).item()
accuracy = (num_correct / len(target_test_loader.dataset)) * 100
print("Accuracy is : {:f}%".format(accuracy))


# In[8]:


all_labels = []
all_pred = []


# In[26]:


num_correct = 0
for images, labels in target_train_loader:
    model.eval()
    images, labels = images.to(DEVICE), labels.to(DEVICE)
    output_class, output_domain = model(images,0)
    pred = torch.argmax(output_class,1)
    num_correct += torch.sum(pred == labels).item()
    
    all_labels += labels.tolist()
    all_pred += pred.tolist()
    
accuracy = (num_correct / len(target_train_loader.dataset)) * 100
print("Accuracy is : {:f}%".format(accuracy))


# In[30]:


accuracies = []
for i in range(11):
    accuracies.append(all_labels.count(i))
#     accuracies.append(all_pred.count(i) /all_labels.count(i))


# In[31]:


accuracies


# In[ ]:


labels = [79.128,53.6,31.53,21.8]
dataset_name = ["Real World", "Product","Art","Clipart"]


# In[ ]:


fig, ax = plt.subplots()
ax.bar(dataset_name,labels)
_ = ax.set_title("Trained on Real World")
ax.set_ylim([0,100])
for i, v in enumerate(labels):
    ax.text(i-.25, 
              v/labels[i]+ 1*labels[i] + 5, 
              labels[i], 
              fontsize=10, 
              )


# In[ ]:


num_correct = 0
for images, labels in target_train_loader:
    model.eval()
    images, labels = images.to(DEVICE), labels.to(DEVICE)
    output_class, output_domain = model(images,0)
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
        pass
    
print(num_correct)
accuracy = (num_correct / len(target_train_loader.dataset)) * 100
print("Accuracy is : {:f}%".format(accuracy))


# In[72]:


list_ground_truth = ground_truth.squeeze().tolist()
list_prediction = prediction.squeeze().tolist()
for element in list_ground_truth:
    if element in list_prediction:
        num_correct+=1


# In[8]:


list_ground_truth


# In[34]:


ground_truth = np.argwhere(labels.cpu()==1)
prediction = np.argwhere(pred.cpu()== 1)

