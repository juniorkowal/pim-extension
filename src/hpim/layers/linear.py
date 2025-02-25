import torch
import torch.nn as nn
from ..ops import mm, add, transpose


class PIMLinear(nn.Module):
    def __init__(self, in_features, out_features, weight=None, bias=None, **kwargs):
        super(PIMLinear, self).__init__()
        self.weight = nn.Parameter(weight if weight is not None else torch.empty((out_features, in_features)))
        self.bias = nn.Parameter(bias if bias is not None else torch.empty((out_features)))

    def forward(self, x):
        # x * A^T + b
        x_at = mm(x, transpose(self.weight))
#        print(x_at.shape, self.bias.shape)
#        if x_at.shape != self.bias.unsqueeze(0).shape:
#           exit()
        return add(x_at, self.bias.unsqueeze(0))
