# -*- coding: utf-8 -*-
import torch


def accuracy(labels, outputs):
    preds = outputs.argmax(-1)
    return (preds == labels.view_as(preds)).float().detach().numpy().mean()


def save_model(model, path):
    torch.save(model.state_dict(), path)
