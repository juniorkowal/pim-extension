from setuptools import setup, find_packages
from torch.utils.cpp_extension import CppExtension, BuildExtension
import os

PIMBLAS_PATH = os.getenv("PIMBLAS_DIR", "/root/pimblas_install")
PIMBLAS_INCLUDE = os.path.join(PIMBLAS_PATH, "include")
PIMBLAS_LIB = os.path.join(PIMBLAS_PATH, "lib")

setup(
    name='hpim',
    version='0.1',
    packages=find_packages(where="src"),
    package_dir={"": "src"},
#    package_data={"hpim": ["*.so"]},
#    include_package_data=True,
    ext_modules=[
        CppExtension(
            name='hpim.hpim',
            sources=[
                'src/extension.cpp',
                'src/ops/mm.cpp',
                'src/ops/add.cpp',
                'src/ops/relu.cpp'
            ],
            include_dirs=[PIMBLAS_INCLUDE, os.path.abspath("src")],
            library_dirs=[PIMBLAS_LIB],
            libraries=["pimblas"],
            extra_compile_args=['-std=c++17', '-std=c++14', '-lstdc++'],
            runtime_library_dirs=[PIMBLAS_LIB],
            extra_link_args=['-L' + PIMBLAS_LIB],
        ),
    ],
    cmdclass={'build_ext': BuildExtension},
    install_requires=[
        'torch',
        'numpy'
    ],
)
