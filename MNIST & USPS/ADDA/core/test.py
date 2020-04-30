"""Test script to classify target data."""

import torch
import torch.nn as nn


def eval_tgt(encoder, classifier, data_loader, DEVICE):
    """Evaluation for target encoder by source classifier on target dataset."""
    # set eval state for Dropout and BN layers
    encoder.eval()
    classifier.eval()

    # init loss and accuracy
    loss = 0
    acc = 0

    # set loss function
    criterion = nn.CrossEntropyLoss()

    # evaluate network
    for (images, labels) in data_loader:
        images = images.to(DEVICE)

        labels = labels.to(DEVICE)
        labels = labels.squeeze()
        # images = torch.cat((images,images,images),1)
        # images = nn.functional.pad(images,(2,2,2,2))

        preds = classifier(encoder(images))
        loss += criterion(preds, labels).item()

        pred_cls = preds.data.max(1)[1]
        acc += pred_cls.eq(labels.data).sum()
    
    loss /= len(data_loader)
    acc = acc.item() / len(data_loader.dataset)

    print("Avg Loss = {}, Avg Accuracy = {:2%}".format(loss, acc))
