from setuptools import setup, find_packages
from torch.utils.cpp_extension import CppExtension, BuildExtension
import os

def package_data():
    return {
        'hpim': ['hpim.so'],
    }

setup(
    name='hpim',
    version='0.1',
    packages=find_packages(),
    ext_modules=[
        CppExtension(
            name='hpim.hpim',
            sources=[
                'src/extension.cpp',
                'src/ops/matmul.cpp',
                'src/ops/add.cpp',
                'src/ops/transpose.cpp'
            ],
        ),
    ],
    package_data=package_data(),
    cmdclass={'build_ext': BuildExtension},
    install_requires=[
        'torch',
        'numpy'
    ],
)
