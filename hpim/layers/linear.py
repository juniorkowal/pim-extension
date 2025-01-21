import torch
import torch.nn as nn
from ..ops import matmul, add, transpose


class PIMLinear(nn.Module):
    def __init__(self, in_features, out_features, weight=None, bias=None):
        super(PIMLinear, self).__init__()
        self.weight = nn.Parameter(weight if weight is not None else torch.empty((out_features, in_features)))
        self.bias = nn.Parameter(bias if bias is not None else torch.empty((out_features)))

    def forward(self, x):
        # x * A^T + b
        return add(matmul(x, transpose(self.weight)), self.bias)
