from setuptools import setup, find_packages
from torch.utils.cpp_extension import CppExtension, BuildExtension
import os

BASE_DIR = os.path.dirname(os.path.realpath(__file__))

PIMBLAS_PATH = os.getenv("PIMBLAS_DIR", "/root/pimblas_install")
PIMBLAS_INCLUDE = os.path.join(PIMBLAS_PATH, "include")
PIMBLAS_LIB = os.path.join(PIMBLAS_PATH, "lib")

packages = find_packages(where='.', include=['torch_hpim', 'torch_hpim.*'])

setup(
    name='torch_hpim',
    version='0.1',
    packages=packages,
    # packages=['torch_hpim', 'torch_hpim.hpim'],
    package_dir={
        'torch_hpim': 'torch_hpim',  # Root package
    },
#    package_data={"hpim": ["*.so"]},
#    include_package_data=True,
    ext_modules=[
        CppExtension(
            name='torch_hpim._C',
            sources=[
                'torch_hpim/csrc/extension.cpp',
                'torch_hpim/csrc/device.cpp',
                'torch_hpim/csrc/ops/mm.cpp',
                'torch_hpim/csrc/ops/add.cpp',
                'torch_hpim/csrc/ops/relu.cpp'
            ],
            include_dirs=[
                PIMBLAS_INCLUDE, 
                os.path.abspath("torch_hpim/csrc"),
                os.path.join(BASE_DIR),
            ],
            library_dirs=[PIMBLAS_LIB],
            libraries=["pimblas"],
            extra_compile_args=['-std=c++17', '-lstdc++'],
            runtime_library_dirs=[PIMBLAS_LIB],
            extra_link_args=['-L' + PIMBLAS_LIB],
        ),
    ],
    cmdclass={'build_ext': BuildExtension},
    install_requires=[
        'torch',
        'numpy'
    ],
    # options={"bdist_wheel": {"universal": True}},
)

# def main():
#     1. upmemsdk # to na kompilacje a tak to dac warning do inita
#     2. libpimblas
#     3 setup()

# if __name__ == "__main__":
#     main()