__all__ = [
    "is_available",
    "device_count",
    "current_device",
    "get_device_name",
]


def is_available():
    return True  # always available for testing

def device_count():
    return 1  # one 'upmem' device

def current_device():
    return 'upmem'  # default device index ?

def get_device_name(device=None):
    return "upmem"
