from setuptools import setup, find_packages
from torch.utils.cpp_extension import CppExtension, BuildExtension
import os


setup(
    name='hpim',
    version='0.1',
    packages=find_packages(),
    ext_modules=[
        CppExtension(
            name='hpim.hpim',
            sources=[
                'src/extension.cpp',
                'src/ops/mm.cpp',
                'src/ops/add.cpp',
                'src/ops/transpose.cpp'
            ],
            include_dirs=[os.path.join(PIMBLAS_PATH, "include")],
            library_dirs=[os.path.join(PIMBLAS_PATH, "lib")],
            libraries=["pimblas"],
            extra_compile_args=['-std=c++14'],
        ),
    ],
    cmdclass={'build_ext': BuildExtension},
    install_requires=[
        'torch',
        'numpy'
    ],
)
