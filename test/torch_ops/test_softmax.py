import pytest
import copy
import torch
import torch_pim

DEVICE = 'upmem'
RTOL = 1e-04

def test_softmax_basic():
    shape = [24,24]
    dim = 1
    x_cpu = torch.randn(shape, dtype=torch.float)
    x_pim = x_cpu.upmem()
    y_pim = torch.nn.functional.softmax(x_pim, dim)
    y_cpu = torch.nn.functional.softmax(x_cpu, dim)
    y_pim = y_pim.cpu()
    assert torch.allclose(y_cpu, y_pim, RTOL) == True

def test_softmax():
    shapes = [
        # (2, 3, 4, 5, 7, 8, 9, 11),
        # (2, 3, 4, 5),
        # (2, 0, 3, 5),
        (2, 3, 4),
        (2, 3),
        # (2,),
    ]
    dtype_list = [(torch.float, 3e-3)#, (torch.half, 3e-3)
                  ]
    for shape in shapes:
        for data_type, err in dtype_list:
            x_cpu = torch.randn(shape, dtype=torch.float)
            x_pim = x_cpu.upmem()
            for dim in range(len(shape)):
                y_pim = torch.nn.functional.softmax(x_pim, dim)
                y_cpu = torch.nn.functional.softmax(x_cpu, dim)
                y_pim = y_pim.cpu()
                assert torch.allclose(y_cpu, y_pim, RTOL) == True

    for data_type, err in dtype_list:
        x_cpu = torch.randn(2, 3, 4, 5, dtype=torch.float)
        x_pim = x_cpu.upmem()
        dims = [-3, -2, -1, 0, 1, 2, 3]
        for i in range(len(dims)):  # pylint: disable=C0200
            y_pim = torch.nn.functional.softmax(x_pim, dims[i])
            y_cpu = torch.nn.functional.softmax(x_cpu, dims[i])
            y_pim = y_pim.cpu()
            assert torch.allclose(y_cpu, y_pim, RTOL) == True

def test_softmax_channels_last():
    shapes = [(2, 3, 4, 5), 
            #   (2, 3, 24, 30), 
            #   (2, 0, 3, 5), 
              (1, 1, 1, 30)
              ]
    dtype_list = [(torch.float, 3e-3)#, (torch.half, 3e-3)
                  ]
    for shape in shapes:
        for data_type, err in dtype_list:
            x_cpu = torch.randn(shape, dtype=torch.float).to(
                memory_format=torch.channels_last
            )
            x_pim = x_cpu.upmem()
            for dim in range(len(shape)):
                y_pim = torch.nn.functional.softmax(x_pim, dim)
                y_cpu = torch.nn.functional.softmax(x_cpu, dim)
                y_pim = y_pim.cpu()
                assert torch.allclose(y_cpu, y_pim, RTOL) == True

    for data_type, err in dtype_list:
        x_cpu = torch.randn(2, 3, 4, 5, dtype=torch.float)
        x_pim = x_cpu.upmem()
        dims = [-3, -2, -1, 0, 1, 2, 3]
        for i in range(len(dims)):  # pylint: disable=C0200
            y_pim = torch.nn.functional.softmax(x_pim, dims[i])
            y_cpu = torch.nn.functional.softmax(x_cpu, dims[i])
            y_pim = y_pim.cpu()
            assert torch.allclose(y_cpu, y_pim, RTOL) == True

# def test_softmax_not_dense():
#     shapes = [(2, 3, 4, 5), 
#               (2, 3, 4), 
#             #   (2, 0, 3, 5), 
#               (2, 3)
#               ]
#     dtype_list = [(torch.float, 3e-3), (torch.half, 3e-3)]
#     for shape in shapes:
#         for data_type, err in dtype_list:
#             x_cpu = torch.randn(shape, dtype=torch.float)
#             x_pim = x_cpu.upmem()
#             for dim in range(len(shape)):
#                 y_pim = torch.nn.functional.softmax(x_pim[:, :2], dim)
#                 y_cpu = torch.nn.functional.softmax(x_cpu[:, :2], dim)
#                 y_pim = y_pim.cpu()
#                 assert torch.allclose(y_cpu, y_pim, RTOL)

def test_nn_Softmax():
    shapes = [
        # (2, 3, 4, 5, 7, 8, 9, 11),
        # (2, 3, 4, 5),
        # (2, 0, 3, 5),
        (2, 3, 4),
        (2, 3),
        # (2,),
    ]
    dtype_list = [(torch.float, 3e-3)#, (torch.half, 3e-3)
                  ]
    for shape in shapes:
        for data_type, err in dtype_list:
            x_cpu = torch.randn(shape, dtype=torch.float, requires_grad=True)
            x_pim = x_cpu.upmem()
            for dim in range(len(shape)):
                model = torch.nn.Softmax(dim=dim)
                model_pim = torch.nn.Softmax(dim=dim).to(DEVICE)
                y_cpu = model(x_cpu)
                y_pim = model_pim(x_pim)
                y_pim = y_pim.cpu()
                assert torch.allclose(y_cpu, y_pim, RTOL) == True

    for data_type, err in dtype_list:
        x_cpu = torch.randn(2, 3, 4, 5, dtype=torch.float)
        x_pim = x_cpu.upmem()
        dims = [-3, -2, -1, 0, 1, 2, 3]
        for i in range(len(dims)):  # pylint: disable=C0200
            y_pim = torch.nn.Softmax(dims[i])(x_pim)
            y_cpu = torch.nn.Softmax(dims[i])(x_cpu)
            y_pim = y_pim.cpu()
            assert torch.allclose(y_cpu, y_pim, RTOL) == True

# def test_nn_Softmax2d():
#     shapes = [(2, 3, 4, 5), 
#             #   (2, 0, 3, 5), 
#               (2, 3, 4)
#               ]
#     dtype_list = [(torch.float, 3e-3), (torch.half, 3e-3)]
#     for shape in shapes:
#         for data_type, err in dtype_list:
#             x_cpu = torch.randn(shape, dtype=torch.float, requires_grad=True)
#             x_pim = x_cpu.upmem()
#             model = torch.nn.Softmax2d()
#             model_pim = torch.nn.Softmax().to(DEVICE)
#             y_cpu = model(x_cpu)
#             y_pim = model_pim(x_pim)
#             y_pim = y_pim.cpu()
#             assert torch.allclose(y_cpu, y_pim, RTOL)

def test_softmax_large():
    shapes = [(128, 128), (1, 128*128)]
    dtype_list = [(torch.float, 3e-3)]
    dim = 1
    for shape in shapes:
        for data_type, err in dtype_list:
            x_cpu = torch.randn(shape, dtype=torch.float)
            x_pim = x_cpu.upmem()
            x = torch.randn(shape, dtype=torch.float)
            out_cpu = x.softmax(dim)
            x_pim = x_cpu.upmem()
            out_pim = x_pim.softmax(dim)
            out_pim = out_pim.cpu()
            assert torch.allclose(out_cpu, out_pim, RTOL) == True
