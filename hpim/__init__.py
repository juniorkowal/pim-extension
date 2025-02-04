def _load_extension():
    import os

    os.environ["LD_LIBRARY_PATH"] = os.getenv("PIMBLAS_DIR", "/root/pimblas_install") + "/lib:" + os.environ.get("LD_LIBRARY_PATH", "")

    import torch

    lib_path = os.path.join(os.path.dirname(__file__), 'hpim.cpython-36m-x86_64-linux-gnu.so')
    torch.ops.load_library(lib_path)

_load_extension()

from .hpim_optimize import optimize