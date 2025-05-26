import pytest
import copy
import numpy as np
import torch
import torch.nn.functional as F
import torch_pim

DEVICE = 'upmem'
RTOL = 1e-07

def test_relu():
    for in_shape in [
        (1),
        (2, 3),
        (8, 224, 224),
        (1, 1, 1, 1),
        (1, 3, 16, 16),
        (1, 3, 16, 16, 3),
    ]:
        input_ = torch.randn(in_shape, dtype=torch.float)
        input_cpu = copy.deepcopy(input_)
        output_cpu = F.relu(input_)
        output_pim = F.relu(input_.upmem())
        output_pim = output_pim.cpu()
        assert torch.allclose(output_cpu, output_pim, RTOL)
        assert torch.allclose(input_cpu, input_, RTOL)

        input_pim = input_.upmem() # test inplace operation
        F.relu(input_cpu, inplace=True)
        F.relu(input_pim, inplace=True)
        input_pim = input_pim.cpu()
        assert torch.allclose(output_cpu, input_pim) == True
        assert torch.allclose(input_cpu, input_pim) == True

# https://github.com/Cambricon/torch_mlu/blob/r2.4_develop/test/torch_ops/test_relu.py
# def test_relu_permute():
#     import random  # pylint: disable=C0415

#     for in_shape in [
#         (8, 224, 224),
#         (1, 1, 1, 1),
#         (1, 3, 16, 16, 4),
#         (1, 3, 16, 16, 3, 6),
#         (1, 3, 16, 16, 4, 15, 8),
#     ]:
#         input_ = torch.randn(in_shape, dtype=torch.float)
#         size = np.arange(len(in_shape))
#         random.shuffle(size)
#         input_pim_ori = input_.to(DEVICE)
#         input_ = torch.permute(input_, tuple(size))
#         input_pim = torch.permute(input_pim_ori, tuple(size))
#         # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#         input_ = input_.cpu()
#         input_pim = input_pim.cpu()
#         # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#         # NOTE: in e.g. torch_mlu deepcopy from device works out of the box
#         # but here we need to transfer to cpu first or we get segfault
#         input_inplace_ = copy.deepcopy(input_)
#         input_pim_inplace_ = copy.deepcopy(input_pim)
#         # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#         input_inplace_ = input_inplace_.upmem()
#         input_pim_inplace_ = input_pim_inplace_.upmem()
#         # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#         output_cpu = F.relu(input_)
#         output_pim = F.relu(input_pim.upmem())
#         F.relu(input_inplace_, inplace=True)  # test inplace operation
#         F.relu(input_pim_inplace_, inplace=True)
#         output_pim = output_pim.cpu()
#         input_pim_inplace_ = input_pim_inplace_.cpu()
#         assert torch.allclose(output_cpu, output_pim, RTOL)
#         assert torch.allclose(input_inplace_, input_pim_inplace_, RTOL)

def test_relu_channels_last():
    for in_shape in [
        (3, 8, 224, 224),
        (1, 1, 1, 1),
        (1, 3, 16, 16),
        (1, 3, 16, 16),
    ]:
        input_ = torch.randn(in_shape, dtype=torch.float).to(
            memory_format=torch.channels_last
        )
        output_cpu = F.relu(input_)
        input_cpu = copy.deepcopy(input_)
        output_pim = F.relu(input_.upmem())
        output_pim = output_pim.cpu()
        assert torch.allclose(output_cpu, output_pim) == True
        assert torch.allclose(input_cpu, input_) == True

        input_cpu = copy.deepcopy(input_).to(memory_format=torch.channels_last)
        input_pim = input_.upmem()  # test inplace operation
        F.relu(input_cpu, inplace=True)
        F.relu(input_pim, inplace=True)
        input_pim = input_pim.cpu()
        assert torch.allclose(output_cpu, input_pim) == True
        assert torch.allclose(input_cpu, input_pim) == True

def test_relu_boundary_value():
    for number in [0, 0.0001, -0.0001, 999999999]:
        x = torch.tensor(number, dtype=torch.float)
        output_cpu = F.relu(x)
        input_cpu = copy.deepcopy(x)
        output_pim = F.relu(x.upmem())
        output_pim = output_pim.cpu()
        assert torch.allclose(output_cpu, output_pim, RTOL)
        assert torch.allclose(input_cpu, x, RTOL)

        input_cpu = copy.deepcopy(x)
        input_pim = x.upmem()  # test inplace operation
        F.relu(input_cpu, inplace=True)
        F.relu(input_pim, inplace=True)
        input_pim = input_pim.cpu()
        assert torch.allclose(output_cpu, input_pim) == True
        assert torch.allclose(input_cpu, input_pim) == True
