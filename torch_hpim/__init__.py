class UpmemModule:
    @staticmethod
    def is_available():
        return True  # always available for testing

    @staticmethod
    def device_count():
        return 1  # one 'upmem' device

    @staticmethod
    def current_device():
        return 'upmem'  # default device index ?

    @staticmethod
    def get_device_name(device=None):
        return "upmem"


def _load_extension():
    import os
    import torch
    import glob
    import torch.utils.backend_registration

    dir_path = os.path.dirname(__file__)
    lib_path = glob.glob(os.path.join(dir_path, '*_C*.so'))
    
    if not lib_path:
        raise FileNotFoundError("_C.so file not found.")
    
    torch.ops.load_library(lib_path[0])

    torch.utils.backend_registration.rename_privateuse1_backend("upmem")
    torch._register_device_module("upmem", UpmemModule) # TODO: this should be like ascend; torch_hpim.hpim instead of UpmemModule
    #torch.utils.generate_methods_for_privateuse1_backend(for_storage=True)

_load_extension()

from .hpim.hpim_optimize import optimize
from .hpim import ops

