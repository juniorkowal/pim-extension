#/bin/bash
export test_devname=upmem
pytest core.py -k test_create_device
pytest core.py -k test_create_tensor_empty_based_on_dev
