import torch
import torch.nn as nn
import torch.nn.functional as F


# Perceptron net
class SimpleNet(nn.Module):
    def __init__(self, D_in, Hidden, D_out):
        super(SimpleNet, self).__init__()
        self.linear_1 = nn.Linear(D_in, Hidden)
        self.linear_2 = nn.Linear(Hidden, D_out)

    def forward(self, x):
        x = F.sigmoid(self.linear_1(x))
        x = self.linear_2(x)
        return x