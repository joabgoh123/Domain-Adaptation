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


def load_dataset(root, name, train_percent, mean=(0,0,0), std=(1,1,1)):
    transformations = transforms.Compose([transforms.ToTensor()])
    dataset = torchvision.datasets.ImageFolder(os.path.join(root,name),transform=transformations)
    classes_dict = {i:data for i,data in enumerate(dataset.classes)}
    train_length = int(len(dataset) * train_percent)
    test_length = len(dataset) - train_length
    train, test = torch.utils.data.random_split(dataset,[train_length,test_length])
    return train,test, classes_dict


# In[4]:


root = "/raid0/students/student17/OfficeHomeDataset_10072016/"
source_name = "Clipart"
target_name = "Art"
source_train, source_test, source_class = load_dataset(root,source_name, 0.8)
target_train, target_test, target_class = load_dataset(root,target_name, 0.8)


# In[5]:


mean_source = (0.5497, 0.5122, 0.4768)
std_source = (0.3168, 0.3110, 0.3199)

mean_target = (0.5497, 0.5122, 0.4768)
std_target = (0.3168, 0.3110, 0.3199)


# In[6]:


source_train, source_test, source_class = load_dataset(root,source_name, 0.9, mean_source, std_source)
target_train, target_test, target_class = load_dataset(root,target_name, 0.8, mean_target, std_target)


# In[7]:


# train_dataset = torch.utils.data.DataLoader(source_train, batch_size=len(source_train), shuffle=True)
# images, samples = iter(train_dataset).__next__()
# print("Train data mean: " + str(images.mean((0,2,3))))
# print("Train data std: " + str(images.std((0,2,3))))
# mean_source = images.mean((0,2,3))
# std_source = images.std((0,2,3))

# train_dataset = torch.utils.data.DataLoader(target_train, batch_size=len(source_train), shuffle=True)
# images, samples = iter(train_dataset).__next__()
# print("Train data mean: " + str(images.mean((0,2,3))))
# print("Train data std: " + str(images.std((0,2,3))))
# mean_target = images.mean((0,2,3))
# std_target = images.std((0,2,3))


# In[8]:


batch_size = 56
num_workers = 8
source_train_loader = torch.utils.data.DataLoader(source_train,batch_size=batch_size, num_workers=num_workers,shuffle=True)
source_test_loader = torch.utils.data.DataLoader(source_test,batch_size=batch_size, num_workers=num_workers,shuffle=True)

target_train_loader = torch.utils.data.DataLoader(target_train,batch_size=batch_size, num_workers=num_workers,shuffle=True)


# In[9]:


resnet50 = torchvision.models.resnet50()
nn.Sequential(*list(resnet50.children())[:-1])


# In[12]:


DEVICE = "cuda:0"
model = Models.ResNet()
# model.load_state_dict(torch.load("C:\\Users\\Joab-PC\\Desktop\\Personal Documents\\Jupyter Notebooks\\DANN_model_RealWorld.pth"))
# model.load_state_dict(torch.load("DANN_model_RealWorld_SGD3.pth"))
model.to(DEVICE)

optimizer = torch.optim.SGD([
        {"params": model.resnet50_features.parameters(), "lr": 0.001},
        {"params": model.class_classifier.parameters(), "lr": 0.01},
        {"params": model.domain_classifier.parameters(), "lr": 0.01},
    ],lr=0.01)


# In[13]:


train_acc_ , target_acc_, train_lost, s_domain_loss, t_domain_loss = Main.train(model,optimizer,source_train_loader,target_train_loader,80,2,DEVICE,batch_size)


# In[17]:


torch.save(model.state_dict(), "DANN_model_Clipart_SGD.pth")


# In[ ]:


images, labels = next(iter(target_train_loader))
images, labels = images.to(DEVICE), labels.to(DEVICE)
output_class, output_domain = model(images,0.1)


# In[ ]:


pred = torch.argmax(output_class,1)


# In[ ]:


pred


# In[ ]:


labels


# In[ ]:


num_correct = 0
pred = torch.argmax(output_class,1)
num_correct += torch.sum(pred == labels).item()
accuracy = (num_correct / 56) * 100
print("Accuracy is : {:f}%".format(accuracy))


# In[15]:


num_correct = 0
for images, labels in target_train_loader:
    model.eval()
    images, labels = images.to(DEVICE), labels.to(DEVICE)
    output_class, output_domain = model(images,0)
    pred = torch.argmax(output_class,1)
    num_correct += torch.sum(pred == labels).item()
accuracy = (num_correct / len(target_train_loader.dataset)) * 100
print("Accuracy is : {:f}%".format(accuracy))


# In[94]:


labels = [79.128,53.6,31.53,21.8]
dataset_name = ["Real World", "Product","Art","Clipart"]


# In[99]:


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


pred = torch.argmax(output_domain,1)
pred


# In[ ]:




