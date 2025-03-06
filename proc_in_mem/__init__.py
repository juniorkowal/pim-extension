def _load_extension():
    import os
    import torch
    import glob

    dir_path = os.path.dirname(__file__)
    lib_path = glob.glob(os.path.join(dir_path, '*hpim*.so'))

    if not lib_path:
        raise FileNotFoundError("hpim.so file not found.")
    
    torch.ops.load_library(lib_path[0])

_load_extension()

from .hpim_optimize import optimize