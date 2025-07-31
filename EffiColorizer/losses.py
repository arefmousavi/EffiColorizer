import torch
from torch import nn


class BCE_GANLoss:
    def __init__(self):
        self.loss = nn.BCEWithLogitsLoss()

    def __call__(self, predictions, label):
        if label == 'real':
            labels = torch.ones_like(predictions)
        elif label == 'fake':
            labels = torch.zeros_like(predictions)
        else:
            raise Exception("not a valid label")

        return self.loss(predictions, labels)
