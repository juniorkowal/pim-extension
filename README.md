# pim-extension README

An experimental PyTorch extension for adding custom C++ operations or layers that can be integrated into PyTorch models.

## Installation

### Using Conda and Pip:
1. Create and activate the environment:
    ```bash
   conda create -n hpim_env python=3.6.3
   conda activate hpim_env
    ```
2.  Install the package:
    ```bash
    pip install .
    ```
    Or for editable mode:
    ```bash
    pip install -e .
    ```

### Using Docker
1. Build the Docker image and run container:
    ```bash
    docker build -t hpim-package .
    docker run -it --rm hpim-package
    ```
2.  Install package inside docker:
    ```bash
    conda activate hpim_env
    pip install .
    ```
    ```bash
    pip install -e .
    ```

## Usage
```python
import torch
import torch.nn as nn
import hpim

class TinyModel(nn.Module):
    def __init__(self):
        super(TinyModel, self).__init__()
        self.linear1 = nn.Linear(100, 200)
        self.activation = nn.ReLU()
        self.linear2 = nn.Linear(200, 10)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        x = self.softmax(x)
        return x

model = TinyModel()
optimized_model = hpim.optimize(model, layers=['linear', 'relu'])
```