import torch
import torch.nn as nn
import torch.nn.functional
# from ..ops import relu


class PIMReLU(nn.Module):
    def __init__(self, inplace):
        super(PIMReLU, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return torch.nn.functional.relu(x, inplace=self.inplace)
