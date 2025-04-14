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



import torch
import torch.utils.backend_registration

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

torch.utils.backend_registration.rename_privateuse1_backend("upmem")
torch._register_device_module("upmem", UpmemModule)
#torch.utils.generate_methods_for_privateuse1_backend(for_storage=True)

