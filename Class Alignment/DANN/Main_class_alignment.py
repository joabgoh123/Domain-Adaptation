import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader as DataLoader
import numpy as np


def compute_train_acc(model, images, labels, alpha, BATCH_SIZE):
    model.eval()
    num_correct = 0
    output_class, output_domain = model(images,0.1)
    pred = torch.argmax(output_class,1)
    num_correct += torch.sum(pred == labels).item()
    accuracy = (num_correct / BATCH_SIZE) * 100
    print("Accuracy is : {:f}%".format(accuracy))
    return accuracy

def compute_target_train_acc(model, images, labels, alpha, BATCH_SIZE):
    model.eval()
    num_correct = 0
    output_class, output_domain = model(images,0.1)
    pred = torch.argmax(output_class,1)
    num_correct += torch.sum(pred == labels).item()
    accuracy = (num_correct / BATCH_SIZE) * 100
    print("Accuracy is : {:f}%".format(accuracy))
    return accuracy



def train(model, optimizer, svhn_images, svhn_labels, mnist_images, N_EPOCH, LOG_INTERVAL, DEVICE, BATCH_SIZE):
    loss_class = torch.nn.CrossEntropyLoss()
    loss_domain = torch.nn.CrossEntropyLoss()
    list_source_acc = []
    list_target_acc = []
    alpha = 0
    for epoch in range(1, N_EPOCH + 1):
        for j in range(100,46000,100):
            random_ints = np.random.choice(46000,100, replace=False)
            svhn_image = svhn_images[random_ints]
            svhn_label = svhn_labels[random_ints]
            mnist_image = mnist_images[random_ints]

            optimizer.zero_grad()
            s_img, s_label = svhn_image.to(DEVICE), svhn_label.to(DEVICE)
            domain_label = torch.zeros(BATCH_SIZE).long().to(DEVICE)
            class_output, domain_output = model(input_data=s_img, alpha=alpha)
            err_s_label = loss_class(class_output, s_label)
            err_s_domain = loss_domain(domain_output, domain_label)

            # Training model using target data
            t_img, t_label = mnist_image.to(DEVICE), svhn_label.to(DEVICE)
    #         t_img = torch.cat((t_img,t_img,t_img),dim=1)
    #         t_img = nn.functional.pad(t_img,(2,2,2,2))
            domain_label = torch.ones(BATCH_SIZE).long().to(DEVICE)
            _, domain_output = model(input_data=t_img, alpha=alpha)
            err_t_domain = loss_domain(domain_output, domain_label)
            err = err_s_label + err_t_domain + err_s_domain

            err.backward()
            optimizer.step()
        
        print("Epoch " + str(epoch))

#         source_acc = compute_train_acc(model, s_img, s_label, alpha, BATCH_SIZE)
#         target_acc = compute_target_train_acc(model, t_img, t_label, alpha, BATCH_SIZE)
#         list_source_acc.append(source_acc)
#         list_target_acc.append(target_acc)
        alpha += 0.01
    return list_source_acc, list_target_acc