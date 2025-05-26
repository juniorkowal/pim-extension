import pytest
import random
import torch
import torch_pim

DEVICE = 'upmem'
RTOL = 1e-07

def test_mm():
    x1 = torch.randn(2, 2)
    x2 = torch.randn(2, 2)
    x1_pim = x1.upmem()
    x2_pim = x2.upmem()
    y_cpu = torch.mm(x1, x2)
    y_pim = torch.mm(x1_pim, x2_pim)
    y_pim = y_pim.cpu()
    assert torch.allclose(y_cpu, y_pim, RTOL) == True

def test_mm_out():
    x1 = torch.randn(2, 2)
    x2 = torch.randn(2, 2)
    x1_pim = x1.upmem()
    x2_pim = x2.upmem()
    y_cpu = torch.zeros(2, 2)
    y_pim = y_cpu.upmem()
    torch.mm(x1, x2, out=y_cpu)
    torch.mm(x1_pim, x2_pim, out=y_pim)
    y_pim = y_pim.cpu()
    assert torch.allclose(y_cpu, y_pim, RTOL) == True

def test_mm_even():
    x1 = torch.randn(8, 8)
    x2 = torch.randn(8, 8)
    x1_pim = x1.upmem()
    x2_pim = x2.upmem()
    y_cpu = torch.mm(x1, x2)
    y_pim = torch.mm(x1_pim, x2_pim)
    y_pim = y_pim.cpu()
    assert torch.allclose(y_cpu, y_pim, RTOL) == True

def test_mm_odd():
    x1 = torch.randn(17, 17)
    x2 = torch.randn(17, 17)
    x1_pim = x1.upmem()
    x2_pim = x2.upmem()
    y_cpu = torch.mm(x1, x2)
    y_pim = torch.mm(x1_pim, x2_pim)
    y_pim = y_pim.cpu()
    assert torch.allclose(y_cpu, y_pim, RTOL) == True

def test_mm_large():
    x1 = torch.randn(128, 128)
    x2 = torch.randn(128, 128)
    x1_pim = x1.upmem()
    x2_pim = x2.upmem()
    y_cpu = torch.mm(x1, x2)
    y_pim = torch.mm(x1_pim, x2_pim)
    y_pim = y_pim.cpu()
    assert torch.allclose(y_cpu, y_pim, RTOL) == True

# def test_mm_int8():
#     ...

# def test_mm_int32():
#     ...

def test_mm_loop():
    num_iter = 10
    mat_range = 300
    for i in range(num_iter):
        rows_mat1 = random.randint(1, mat_range)
        cols_mat1 = random.randint(1, mat_range)
        rows_mat2 = cols_mat1
        cols_mat2 = random.randint(1, mat_range)
        mat1 = torch.randn(rows_mat1, cols_mat1)
        mat2 = torch.randn(rows_mat2, cols_mat2)
        res_torch = torch.mm(mat1, mat2)
        mat1_pim = mat1.upmem()
        mat2_pim = mat2.upmem()
        res_pim = torch.mm(mat1_pim, mat2_pim)
        res_pim = res_pim.cpu()
        assert torch.isnan(res_pim).any() == False
        assert torch.allclose(res_pim, res_torch, RTOL) == True