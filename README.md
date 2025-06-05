# PyTorch PIM Extension

## Overview
An experimental PyTorch extension providing UPMEM PIM hardware acceleration through custom C++ operators built on [PIMBLAS](https://github.com/3cityrnd/libpimblas/tree/main) kernels. Delivers torch-npu/torch_mlu-style device integration via PyTorch's `PrivateUse1` dispatch key, currently supporting core ML operations (Softmax, ReLu, mm, add, ...). Two approaches available: (1) main branch implements native device backend, while (2) [`stable/hpim/2.7`](https://github.com/your-repo/tree/stable/hpim/2.7) uses operator replacement for transparent PIM optimization without full device implementation.

**Status**: Experimental  
**Stable Branch**: [`stable/hpim/2.7`](https://github.com/your-repo/tree/stable/hpim/2.7) (operator replacement technique)

## Installation

### Prerequisites
- **OS**: Supported distributions:
  | Distribution | Version | SDK default Path               |
  |--------------|---------|------------------------|
  | Debian       | 10      | `debian10/upmem-2025.1.0-Linux-x86_64` |
  | Ubuntu       | 20.04   | `ubuntu-2004/...`     |
  | Ubuntu       | 22.04   | `ubuntu-2204/...`     |
  | Rocky Linux  | 8/9.2   | `rocky8/...`          |
- **UPMEM SDK**: [2025.1.0](https://sdk.upmem.com/2025.1.0/)
- **PyTorch**: 2.0+

### Build from Source
```bash
git clone https://github.com/juniorkowal/pim-extension.git
cd pim-extension

python setup.py install
```
Note: Simulator mode is automatically used when PIM hardware is unavailable.

## Getting Started
### Environment Setup
```bash
# Required before execution
source $UPMEM_SDK_DIRECTORY/upmem_env.sh
# Or if built from source and upmemsdk python package is available
source `python -m upmemsdk`
```
### Basic Usage
```bash
import torch
import torch.nn as nn
import torch_pim
import torch.nn.functional as F
torch.manual_seed(0)

class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()
        self.linear1 = nn.Linear(100, 200)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        return x

if __name__ == "__main__":
    DEVICE = 'upmem'

    model = BaseModel().to(DEVICE)
    input_data = torch.randn(1, 100).to(DEVICE)
    
    model.eval()

    with torch.no_grad():
        output = model(input_data)

    print("Output: ", output.to('cpu')) # Transfer to CPU for output
```

## Verification Test
```bash
pip install pytest
pytest test/
```

## Development Resources
- [PyTorch Custom C++ Extensions](https://pytorch.org/tutorials/advanced/cpp_extension.html)  
- [Extending Dispatcher for New Backends](https://pytorch.org/tutorials/advanced/extend_dispatcher.html)  
- [PrivateUse1 Backend Integration](https://pytorch.org/tutorials/advanced/privateuseone.html)  
- [PyTorch TensorIterator Internals](https://labs.quansight.org/blog/2021/04/pytorch-tensoriterator-internals-update)  
- [Ascend PyTorch Adapter](https://github.com/Ascend/pytorch)  
- [Cambricon Torch_MLU](https://github.com/Cambricon/torch_mlu/tree/r2.4_develop)  
- [UPMEM PIMBLAS Kernels](https://github.com/3cityrnd/libpimblas/tree/main)  
- [Torch.fx Explained](https://www.youtube.com/watch?v=5FNHwPIyHr8)  
- [PyTorch Dispatcher Deep Dive](https://blog.ezyang.com/2020/09/lets-talk-about-the-pytorch-dispatcher/)  
- [Open Registration Extension Example](https://github.com/pytorch/pytorch/blob/main/test/cpp_extensions/open_registration_extension.cpp)  
- [Operator Decomposition Tests](https://github.com/pytorch/pytorch/blob/main/test/jit/test_op_decompositions.py)  
- [PyTorch C++ API Docs](https://pytorch.org/cppdocs/)  
- [ATen Native Functions](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/README.md)  
