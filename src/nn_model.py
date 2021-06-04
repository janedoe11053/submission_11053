import torch
import torch.nn as nn
import torch.nn.functional as f


class NetModel(nn.Module):
    def __init__(self, features, neurons):
        super().__init__()
        self.lin1 = nn.Linear(features, neurons)
        self.b1 = nn.BatchNorm1d(neurons)
        self.lin2 = nn.Linear(neurons, 8)
        self.lin3 = nn.Linear(8, 1)

    def forward(self, x):
        x = x.float()
        x = self.b1(self.lin1(x))
        x = f.relu(self.lin2(x))
        x = torch.sigmoid(self.lin3(x))
        return x
