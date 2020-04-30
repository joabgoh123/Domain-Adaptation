#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader as DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt
import os
import Main
import Models
import utils
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


source_train_loader, source_test_loader = utils.load_data("SVHN")
target_train_loader, target_test_loader = utils.load_data("MNIST")
batch_size = 64


# In[3]:


target_train_loader.dataset


# In[4]:


images, labels = next(iter(target_train_loader))
images.size()


# In[5]:


DEVICE = "cuda"
model = Models.DANN()
# model.load_state_dict(torch.load("C:\\Users\\Joab-PC\\Desktop\\Personal Documents\\Jupyter Notebooks\\DANN_model.pth"))
model.to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(),lr=0.001)
model


# In[6]:


source_acc, target_acc = Main.train(model,optimizer,source_train_loader,target_train_loader,30,400,DEVICE,batch_size)


# In[28]:


torch.save(model.state_dict(), "C:\\Users\\Joab-PC\\Desktop\\Personal Documents\\Jupyter Notebooks\\DANN_LeNet5_4.pth")


# In[14]:


images, labels = next(iter(source_train_loader))
images, labels = images.to(DEVICE), labels.to(DEVICE)
output_class, output_domain = model(images,0.1)


# In[15]:


num_correct = 0
pred = torch.argmax(output_class,1)
num_correct += torch.sum(pred == labels).item()
accuracy = (num_correct / 56) * 100
print("Accuracy is : {:f}%".format(accuracy))


# In[16]:


num_correct = 0
for images, labels in source_test_loader:
    images, labels = images.to(DEVICE), labels.to(DEVICE)
    output_class, output_domain = model(images,0.1)
    pred = torch.argmax(output_class,1)
    num_correct += torch.sum(pred == labels).item()
accuracy = (num_correct / len(source_test_loader.dataset)) * 100
print("Accuracy is : {:f}%".format(accuracy))


# In[17]:


pred = torch.argmax(output_domain,1)
pred


# In[18]:


def compute_accuracy(device):
    num_correct = 0
    for images,labels in target_test_loader:
        images = images.to(device)
        labels = labels.to(device)
        images = torch.cat((images,images,images),dim=1)
        images = nn.functional.pad(images,(2,2,2,2))
        output, _ = model(images,0.1)
        pred = torch.argmax(output,1)
        num_correct += torch.sum(pred == labels).item()
    accuracy = (num_correct / len(target_test_loader.dataset)) * 100
    print("Accuracy is : {:f}%".format(accuracy))


# In[25]:


compute_accuracy(DEVICE)


# In[27]:


fig, ax = plt.subplots()
_ = ax.set_title ("SVHN -> MNIST")
adaptive_loss = ax.plot(source_acc,label="Source (SVHN)")
classification_loss = ax.plot(target_acc, label="Target (MNIST)")
_ = ax.set_ylabel("Training Accuracy")
_ = ax.set_xlabel("Epochs")
_ = plt.legend()


# In[ ]:




