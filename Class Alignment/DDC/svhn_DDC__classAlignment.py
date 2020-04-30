#!/usr/bin/env python
# coding: utf-8

# In[6]:


import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets,transforms
import torch.nn as nn
import torch.nn.functional as F
import time
import functools
import utils
import DDC
from collections import OrderedDict
from torch.autograd import Variable
import random
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


train_loader_svhn, test_loader_svhn = utils.load_data("SVHN")
train_loader_mnist, test_loader_mnist = utils.load_data("MNIST")
svhn_images = torch.load('svhn_data.pt')
svhn_labels = torch.load('svhn_labels.pt')
svhn_labels = svhn_labels.type(torch.long)
mnist_images = torch.load('mnist_data.pt')

mnist_images = torch.cat((mnist_images,mnist_images,mnist_images),dim=1)
mnist_images = nn.functional.pad(mnist_images,(2,2,2,2))


# In[3]:


class LeNet5(nn.Module):
    """
    Input - 1x32x32
    C1 - 6@28x28 (5x5 kernel)
    tanh
    S2 - 6@14x14 (2x2 kernel, stride 2) Subsampling
    C3 - 16@10x10 (5x5 kernel, complicated shit)
    tanh
    S4 - 16@5x5 (2x2 kernel, stride 2) Subsampling
    C5 - 120@1x1 (5x5 kernel)
    F6 - 84
    tanh
    F7 - 10 (Output)
    """
    def __init__(self):
        super(LeNet5, self).__init__()
        self.convnet = nn.Sequential(OrderedDict([
            ('c1', nn.Conv2d(3, 6, kernel_size=(5, 5))),
            ('relu1', nn.ReLU()),
            ('s2', nn.MaxPool2d(kernel_size=(2, 2), stride=2)),
            ('c3', nn.Conv2d(6, 16, kernel_size=(5, 5))),
            ('relu3', nn.ReLU()),
            ('s4', nn.MaxPool2d(kernel_size=(2, 2), stride=2)),
            ('c5', nn.Conv2d(16, 120, kernel_size=(5, 5))),
            ('relu5', nn.ReLU())
        ]))

        self.fc = nn.Sequential(OrderedDict([
            ('f6', nn.Linear(120, 84)),
            ('relu6', nn.ReLU()),
        ]))

        self.fc_cls = nn.Sequential(OrderedDict([
            ('f7', nn.Linear(84, 10))
        ]))
    
    def forward(self, x):
        x = self.convnet(x)
        x = x.view(-1,120)
        x = self.fc(x)
        x = self.fc_cls(x)
        return x

device = 'cuda:0'
model = LeNet5()
model.to(device)
model.load_state_dict(torch.load('SVHN_LeNet5.pth'))


# In[4]:


get_ipython().run_cell_magic('time', '', '#Calculate Accuracy\nmodel.eval()\nnum_correct = 0\nfor images,labels in train_loader_mnist:\n    images = images.to(device)\n    labels = labels.to(device)\n    images = torch.cat((images,images,images),dim=1)\n    images = nn.functional.pad(images,(2,2,2,2))\n        \n    output = model(images)\n    pred = torch.argmax(output,1)\n    num_correct += torch.sum(pred == labels).item()\nprint("Test accuracy: " + str(num_correct/len(train_loader_mnist.dataset)*100 ) + "%")')


# In[14]:


tensors = [[] for i in range(10)]
for i in range(len(mnist_images)):
    img = mnist_images[i].unsqueeze(dim=0)
    img = img.to(device)
    output = model(img)
    pred = torch.argmax(output,1).item()
    tensors[pred].append(i)
    print(i)


# In[ ]:


# tensors = [[] for i in range(10)]
# for images, labels in train_loader_mnist:
#     model.eval()
#     images = torch.cat((images,images,images),dim=1)
#     images = nn.functional.pad(images,(2,2,2,2))
#     images = images.to(device)
    
#     output = model(images)
#     pred = torch.argmax(output,1)
#     for i in range(len(images)):
#         tensors[pred[i]].append(images[i])


# In[20]:


#Load model
model = DDC.DDC()
model.to(device)

#Load pretrained
# model.load_state_dict(torch.load('svhn_coral.pth'))


# In[21]:


#Define Optimizer
LEARNING_RATE = 0.001
MOMENTUM = 0.9
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam([{'params': model.convNet.parameters(), 'lr':LEARNING_RATE},
                             {'params': model.fc.parameters(), 'lr':LEARNING_RATE*10}],
                             lr=LEARNING_RATE)


# In[22]:


def retrieve_mnist_img_idx(labels):
    img_indexs = []
    for label in labels.tolist():
        img_indexs.append(random.choice(tensors[label]))
    return img_indexs


# In[23]:


#Loss Curves
list_mmd_loss = []
list_class_loss = []
#Training
epochs = 10
for i in range(epochs):
    coral_epoch = 0
    class_loss = 0
    _lambda = 1
    epoch_loss = 0
    start_time = time.time()
    model.train()
    prev_j = 0
    for j in range(100,46000,100):
        random_ints = np.random.choice(46000,100, replace=False)
        svhn_image = svhn_images[random_ints].to(device)
        svhn_label = svhn_labels[random_ints].to(device)
#         mnist_index = retrieve_mnist_img_idx(svhn_label)
        mnist_image = mnist_images[random_ints].to(device)
        
        optimizer.zero_grad()
        #Forward pass
        out_source, out_target = model(svhn_image,mnist_image)
        
        pred = torch.argmax(out_source,1)
        #Calculate loss
        classification_loss =  criterion(out_source, svhn_label)
    #         classification_loss = criterion(out_source,source_label)
        coral_loss = DDC.mmd_linear(out_source,out_target)
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
    print("MMD loss " + str(coral_epoch))
    print("class loss " + str(class_loss))
    print("Epoch {:d} completed, time taken: {:f}".format(i+1,time_taken),end="\t")
    print("Training Loss: " + str(epoch_loss))
    list_mmd_loss.append(coral_loss)
    list_class_loss.append(class_loss)
 
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
    


# In[15]:


tensors = [[] for i in range(10)]
for i in range(len(mnist_images)):
    img = mnist_images[i].unsqueeze(dim=0)
    img = img.to(device)
    output, _ = model(img,img)
    pred = torch.argmax(output,1).item()
    tensors[pred].append(i)
    print(i)


# In[24]:


get_ipython().run_cell_magic('time', '', '#Calculate Accuracy\nmodel.eval()\nnum_correct = 0\nfor images,labels in test_loader_mnist:\n    images = images.to(device)\n    labels = labels.to(device)\n    images = torch.cat((images,images,images),dim=1)\n    images = nn.functional.pad(images,(2,2,2,2))\n        \n    output, _ = model(images,images)\n    pred = torch.argmax(output,1)\n    num_correct += torch.sum(pred == labels).item()\nprint("Test accuracy: " + str(num_correct/len(test_loader_mnist.dataset)*100 ) + "%")')


# In[12]:


get_ipython().run_cell_magic('time', '', '#Calculate Accuracy\nmodel.eval()\nnum_correct = 0\nfor images,labels in test_loader_svhn:\n    images = images.to(device)\n    labels = labels.to(device)\n        \n    output, _ = model(images,images)\n    pred = torch.argmax(output,1)\n    num_correct += torch.sum(pred == labels).item()\nprint("Test accuracy: " + str(num_correct/len(test_loader_svhn.dataset)*100 ) + "%")')


# In[27]:


#Plot losses
plt.plot(list_class_loss, 'orange')
plt.plot(list_mmd_loss)
plt.xticks(np.arange(1, epochs))
plt.title("DDC (Class Alignment)")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend(["Classification loss","Adaptive Loss"])


# In[ ]:




