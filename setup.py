from setuptools import setup, find_packages
from torch.utils.cpp_extension import CppExtension, BuildExtension
import os

def find_package_data(root):
    data = []
    for dirpath, _, filenames in os.walk(root):
        for filename in filenames:
            if filename.endswith('.py'):
                data.append(os.path.relpath(os.path.join(dirpath, filename), root))
    return data

setup(
    name='hpim',
    version='0.1',
    packages=['hpim'],
    package_dir={'hpim': 'proc_in_mem'},
    ext_modules=[
        CppExtension(
            name='hpim.hpim',
            sources=[
                'src/extension.cpp',
                'src/ops/mm.cpp',
                'src/ops/add.cpp',
                'src/ops/transpose.cpp'
            ],
        ),
    ],
    cmdclass={'build_ext': BuildExtension},
    install_requires=[
        'torch',
        'numpy'
    ],
    include_package_data=True,
    package_data={
        'hpim': ['*.so'] + find_package_data('proc_in_mem'),
    },
)