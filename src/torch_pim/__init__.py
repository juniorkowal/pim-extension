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
    lib_path = glob.glob(os.path.join(dir_path, "*_C*.so"))
    
    if not lib_path:
        raise FileNotFoundError("_C.so file not found.")
    if os.environ.get("UPMEM_HOME") is None:
        print("UPMEM_HOME not found, please install and source upmemsdk first.")
        print("If you installed from source, you can use: source `python -m upmemsdk`")
        raise RuntimeError("UPMEM_HOME not found.")

    kernel_dir = os.path.join(dir_path, "kernels")
    if not os.path.exists(kernel_dir):
        raise FileNotFoundError(f"Kernel directory not found at: {kernel_dir}")
    
    kernel_files = glob.glob(os.path.join(kernel_dir, "*.kernel"))
    if not kernel_files:
        raise FileNotFoundError(f"No kernel files (*.kernel) found in: {kernel_dir}")
    
    os.environ["PIMBLAS_KERNEL_DIR"] = str(kernel_dir)

    torch.ops.load_library(lib_path[0])

    torch.utils.backend_registration.rename_privateuse1_backend("upmem")
    torch._register_device_module("upmem", UpmemModule) # TODO: change dummy UpmemModule to our extension folder torch_pim.pim
    #torch.utils.generate_methods_for_privateuse1_backend(for_storage=True)

_load_extension()

from .pim.pim_optimize import optimize
from .pim import ops

