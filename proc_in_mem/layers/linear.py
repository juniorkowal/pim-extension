import torch
import torch.nn as nn
from ..ops import mm, add


class PIMLinear(nn.Module):
    def __init__(self, in_features, out_features, weight=None, bias=None, **kwargs):
        super(PIMLinear, self).__init__()
        self.weight = nn.Parameter(weight if weight is not None else torch.empty((out_features, in_features)))
        self.bias = nn.Parameter(bias if bias is not None else torch.empty((out_features)))

    def forward(self, x):
        # x * A^T + b
        return add(mm(x, self.weight.t().contiguous()), self.bias)
