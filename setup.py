from setuptools import setup, find_packages
from torch.utils.cpp_extension import CppExtension, BuildExtension
import os

PIMBLAS_PATH = os.getenv("PIMBLAS_DIR", "/root/pimblas_install")

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
            include_dirs=[os.path.join(PIMBLAS_PATH, "include")],
            library_dirs=[os.path.join(PIMBLAS_PATH, "lib")],
            libraries=["pimblas"],
            extra_compile_args=['-std=c++14'],
        ),
    ],
    package_data=package_data(),
    cmdclass={'build_ext': BuildExtension},
    install_requires=[
        'torch',
        'numpy'
    ],
)
