import pytest
import torch
import torch_pim

DEVICE = 'upmem'

def test_device():
    assert torch.upmem.is_available() == True

def test_empty_tensor():
    empty = torch.empty(size=[2,2], device=DEVICE)
    assert empty.is_upmem == True

def test_cpu_to_device():
    empty = torch.empty(size=[2,2], device='cpu')
    empty = empty.to(DEVICE)
    assert empty.is_upmem == True

def test_device_to_cpu():
    empty = torch.empty(size=[2,2], device=DEVICE)
    empty = empty.to('cpu')
    assert empty.is_cpu == True
