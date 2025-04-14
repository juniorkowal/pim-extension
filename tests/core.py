import pytest
import os
import torch
import hpim


#
#class UpmemModule:
#        @staticmethod
#        def is_available():
#            return True  # Always available for demonstration purposes
#
#        @staticmethod
#        def device_count():
#               return 1  # Only one 'upmem' device
#        @staticmethod
#        def current_device():
#              return 0  # Default device index
#        @staticmethod
#        def get_device_name(device=None):
#               return "upmem"
#

def info(*args,**kwargs):
    print("[info] ",  *args,**kwargs)


def get_env(name,name_default):
    n=os.getenv(name,name_default)
    info(f'get_env()  {name}={n}   default={name_default}')
    return n

def test_load_torch_show_version():
    info(torch.__version__)
    assert 1


def test_load_hpim_show_version():
   
    info('hpim unknown version???')
    assert 1

def test_create_device():
    dev_name=get_env('test_devname','pim')
    dev=torch.device(dev_name)

def test_create_tensor_empty_based_on_dev():
     #torch.utils.rename_privateuse1_backend("upmem")
     #torch._register_device_module("upmem", UpmemModule)
     #torch.utils.generate_methods_for_privateuse1_backend(for_storage=True)
     e=torch.empty(2,2,device=get_env('test_devname','pim'))



