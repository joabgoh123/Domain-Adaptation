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
import Main_class_alignment
import Models
import numpy as np
import utils
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


DEVICE = "cuda"
model = Models.DANN()
model.to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(),lr=0.001)
model


# In[4]:


loss_class = torch.nn.CrossEntropyLoss()
loss_domain = torch.nn.CrossEntropyLoss()
alpha = 0.001
for epoch in range(1, 10 + 1):
    for j in range(100,46000,100):
        model.train()
#         p = float(epoch * 100) / 10 / 100
#         alpha = 2. / (1. + np.exp(-10 * p)) - 1
        random_ints = np.random.choice(46000,100, replace=False)
        svhn_image = svhn_images[random_ints]
        svhn_label = svhn_labels[random_ints]
        mnist_image = mnist_images[random_ints]

        optimizer.zero_grad()
        s_img, s_label = svhn_image.to(DEVICE), svhn_label.to(DEVICE)
        domain_label = torch.zeros(100).long().to(DEVICE)
        class_output, domain_output = model(input_data=s_img, alpha=alpha)
        err_s_label = loss_class(class_output, s_label)
        err_s_domain = loss_domain(domain_output, domain_label)

#         Training model using target data
        t_img, t_label = mnist_image.to(DEVICE), svhn_label.to(DEVICE)
        domain_label = torch.ones(100).long().to(DEVICE)
        _, domain_output = model(input_data=t_img, alpha=alpha)
        err_t_domain = loss_domain(domain_output, domain_label)
        err = err_s_label + err_t_domain + err_s_domain

#         err_s_label.backward()
        err.backward()
        optimizer.step()
        alpha += 0.001

    print("Epoch " + str(epoch))


# In[5]:


# batch_size = 100
# source_acc, target_acc = Main_class_alignment.train(model,optimizer,svhn_images,svhn_labels,mnist_images,10,400,DEVICE,batch_size)


# In[6]:


num_correct = 0
for images, labels in test_loader_svhn:
    images, labels = images.to(DEVICE), labels.to(DEVICE)
    model.eval()
    output_class, output_domain = model(images,0.1)
    pred = torch.argmax(output_class,1)
    num_correct += torch.sum(pred == labels).item()
accuracy = (num_correct / len(test_loader_svhn.dataset)) * 100
print("Accuracy is : {:f}%".format(accuracy))


# In[7]:


def compute_accuracy(device):
    num_correct = 0
    for images,labels in test_loader_mnist:
        model.eval()
        images = images.to(device)
        labels = labels.to(device)
        images = torch.cat((images,images,images),dim=1)
        images = nn.functional.pad(images,(2,2,2,2))
        output, _ = model(images,0.1)
        pred = torch.argmax(output,1)
        num_correct += torch.sum(pred == labels).item()
    accuracy = (num_correct / len(test_loader_mnist.dataset)) * 100
    print("Accuracy is : {:f}%".format(accuracy))



compute_accuracy(DEVICE)




fig, ax = plt.subplots()
_ = ax.set_title ("SVHN -> MNIST")
adaptive_loss = ax.plot(source_acc,label="Source (SVHN)")
classification_loss = ax.plot(target_acc, label="Target (MNIST)")
_ = ax.set_ylabel("Training Accuracy")
_ = ax.set_xlabel("Epochs")
_ = plt.legend()


