import torch
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

# def compute_target_train_acc(model, images, labels, alpha, BATCH_SIZE):
#     model.eval()
#     num_correct = 0
#     output_class, output_domain = model(images,0.1)
#     pred = torch.argmax(output_class,1)
#     num_correct += torch.sum(pred == labels).item()
#     accuracy = (num_correct / BATCH_SIZE) * 100
#     print("Accuracy is : {:f}%".format(accuracy))
#     return accuracy


def train(model, optimizer, dataloader_src, dataloader_tar, N_EPOCH, LOG_INTERVAL, DEVICE, BATCH_SIZE):
    loss_class = torch.nn.CrossEntropyLoss()
    loss_domain = torch.nn.CrossEntropyLoss()
    list_source_acc = []
    list_target_acc = []
    list_train_lost = []
    list_s_domain = []
    list_t_domain = []
    for epoch in range(1, N_EPOCH + 1):
        model.train()
        len_dataloader = min(len(dataloader_src), len(dataloader_tar))
        data_src_iter = iter(dataloader_src)
        data_tar_iter = iter(dataloader_tar)

        i = 0
        while i < len_dataloader-1:
            p = float(i + epoch * len_dataloader) / N_EPOCH / len_dataloader
#             alpha = 2. / (1. + np.exp(-10 * p)) - 1
            alpha = 0 
            # Training model using source data
            data_source = data_src_iter.next()
            optimizer.zero_grad()
            s_img, s_label = data_source[0].to(DEVICE), data_source[1].to(DEVICE)
            domain_label = torch.zeros(BATCH_SIZE).long().to(DEVICE)
            class_output, domain_output = model(input_data=s_img, alpha=alpha)
            err_s_label = loss_class(class_output, s_label)
            err_s_domain = loss_domain(domain_output, domain_label)

            # Training model using target data
            data_target = data_tar_iter.next()
            t_img, t_label = data_target[0].to(DEVICE), data_target[1].to(DEVICE)
            domain_label = torch.ones(BATCH_SIZE).long().to(DEVICE)
            _, domain_output = model(input_data=t_img, alpha=alpha)
            err_t_domain = loss_domain(domain_output, domain_label)
            err = err_s_label + err_t_domain + err_s_domain
            alpha+=0.01
            err.backward()
            optimizer.step()

            if i % LOG_INTERVAL == 0:
                print(
                    'Epoch: [{}/{}], Batch: [{}/{}], err_s_label: {:.4f}, err_s_domain: {:.4f}, err_t_domain: {:.4f}'.format(
                        epoch, N_EPOCH, i, len_dataloader-1, err_s_label.item(), err_s_domain.item(),
                        err_t_domain.item()))
            i += 1
        source_acc = compute_train_acc(model, s_img, s_label, alpha, BATCH_SIZE)
        target_acc = compute_train_acc(model, t_img, t_label, alpha, BATCH_SIZE)
        list_source_acc.append(source_acc)
        list_target_acc.append(target_acc)
        
        list_train_lost.append(err_s_label.item())
        list_s_domain.append(err_s_domain.item())
        list_t_domain.append(err_t_domain.item())
       

    return list_source_acc, list_target_acc, list_train_lost, list_s_domain, list_t_domain