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
import utils
import coral
from torch.autograd import Variable
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


#Function to display images
def showImage(images,labels,title="SVHN Dataset"):
    if title.upper() == "MNIST":
        images = torch.cat((images,images,images),dim=1)
        images = nn.functional.pad(images,(2,2,2,2))
    rows = 2
    cols = 5
    k = 0
    if type(labels) == torch.Tensor:
        labels = np.array(labels).tolist()
    #Un-normalize
    img_to_display = images[:10].permute(0,2,3,1).cpu()
    labels = labels[:10]
    fig, ax = plt.subplots(rows,cols,figsize=(10, 5))
    for i in range(rows):
        for j in range(cols):
            ax[i][j].imshow(img_to_display[k])
            _ = ax[i][j].set(xlabel = labels[k])
            k+=1
    _ = fig.suptitle(title)


# In[3]:


device = 'cuda:0'


# In[4]:


train_loader_svhn, test_loader_svhn = utils.load_data("SVHN")
train_loader_mnist, test_loader_mnist = utils.load_data("MNIST")


# In[5]:


#Display svhn
images,labels = iter(test_loader_svhn).__next__()
showImage(images,labels)


# In[6]:


images,labels = iter(test_loader_mnist).__next__()
showImage(images,labels,"MNIST")


# In[10]:


#Load model
model = coral.DeepCORAL()
model.to(device)

#Load pretrained
# pretrain_dict = torch.load("C:\\Users\\Joab-PC\\Desktop\\Personal Documents\\Jupyter Notebooks\\svhn_model_statedict.pth")
# model.convNet.state_dict().update(pretrain_dict)
model.load_state_dict(torch.load('svhn_coral.pth'))


# In[8]:


#Define Optimizer
LEARNING_RATE = 0.001
MOMENTUM = 0.9
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam([{'params': model.convNet.parameters(), 'lr':LEARNING_RATE},
                             {'params': model.fc.parameters(), 'lr':LEARNING_RATE*10}],
                             lr=LEARNING_RATE)


# In[9]:


source, target = list(enumerate(train_loader_svhn)), list(enumerate(train_loader_mnist))
train_steps = min(len(source), len(target))


# In[10]:


_, (source_data, source_label) = source[1]
_, (target_data, _) = target[1]
target_data = torch.cat((target_data,target_data,target_data),dim=1)
target_data = nn.functional.pad(target_data,(2,2,2,2))
out1, out2 = model(source_data.to(device),target_data.to(device))
coral.coral_loss(out1,out2)


# In[11]:


#Training
epochs = 30
for i in range(epochs):
    coral_epoch = 0
    class_loss = 0
    epoch_loss = 0
    _lambda = 0.05
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
        target_data = torch.cat((target_data,target_data,target_data),dim=1)
        target_data = nn.functional.pad(target_data,(2,2,2,2))
        
        
        
        optimizer.zero_grad()
        #Forward pass
        out_source, out_target = model(source_data,target_data)
        #Calculate loss
        classification_loss =  torch.nn.functional.cross_entropy(out_source, source_label)
#         classification_loss = criterion(out_source,source_label)
        coral_loss = coral.coral_loss(out_source,out_target)
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
    


# In[11]:


get_ipython().run_cell_magic('time', '', '#Calculate Accuracy\nmodel.eval()\nnum_correct = 0\nfor images,labels in test_loader_svhn:\n    images = images.to(device)\n    labels = labels.to(device)\n    output,_ = model(images,images)\n    pred = torch.argmax(output,1)\n    num_correct += torch.sum(pred == labels).item()\nprint("Test accuracy: " + str(num_correct/len(test_loader_svhn.dataset)*100 ) + "%")')


# In[12]:


get_ipython().run_cell_magic('time', '', '#Calculate Accuracy\nmodel.eval()\nnum_correct = 0\nfor images,labels in test_loader_mnist:\n    images = images.to(device)\n    labels = labels.to(device)\n    images = torch.cat((images,images,images),dim=1)\n    images = nn.functional.pad(images,(2,2,2,2))\n        \n    output,_ = model(images,images)\n    pred = torch.argmax(output,1)\n    num_correct += torch.sum(pred == labels).item()\nprint("Test accuracy: " + str(num_correct/len(test_loader_mnist.dataset)*100 ) + "%")')


# In[14]:


# torch.nn.init.xavier_uniform(model.fc.weight)


# In[15]:


#Show predictions
images, labels = iter(test_loader_svhn).__next__()
images = images.to(device)
output,_ = model(images,images)
pred = torch.argmax(output,1)
label_list = np.array(labels.cpu()).tolist()
pred_list = np.array(pred.cpu()).tolist()
xlabels = []
for i in range(len(label_list)):
    temp_label = "Actual: " + str(label_list[i]) + " Predicted: " + str(pred_list[i])
    xlabels.append(temp_label)

showImage(images,xlabels)


# In[16]:


torch.save(model.state_dict(), "svhn_coral.pth")


# In[ ]:




